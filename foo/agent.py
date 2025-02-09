import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type

import requests
from agentdesk import Desktop
from devicebay import Device
from orign import ReplayBuffer, V1ChatEvent, V1MSSwiftBufferParams
from orign.config import GlobalConfig
from orign.models import ChatResponse
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks.reviewable import AnnotationReviewable
from surfkit.agent import TaskAgent
from surfkit.skill import Skill
from taskara import Task, TaskStatus
from taskara.server.models import V1TaskUpdate
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
)
from toolfuse.util import AgentUtils

from .actor.base import Actor, Step
from .actor.orign import OrignActor

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)


class FooConfig(BaseModel):
    pass


class Foo(TaskAgent):
    """A desktop agent that learns"""

    def learn_task(
        self,
        task: Task,
        skill: Skill,
    ):
        """Learn a task

        Args:
            task (Task): The task
            skill (Skill): The associated skill
        """
        print("learning task: ", task.id)

        if not task.remote or not task.auth_token:
            raise ValueError("Task remote or token not set")

        actor_adapter = f"{skill.name}-actor"
        val_adapter = f"{skill.name}-val"

        orign_config = GlobalConfig(api_key=task.auth_token)

        actor_sft_buffer = ReplayBuffer(
            name=actor_adapter,
            vram_request="40Gi",
            dtype="bfloat16",
            train_every=50,
            sample_n=100,
            sample_strategy="Random",
            ms_swift_params=V1MSSwiftBufferParams(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                model_type="qwen2_5_vl",
                train_type="lora",
                deepspeed="zero3",
                torch_dtype="bfloat16",
                max_length=16384,
                val_split_ratio=1.0,
                num_train_epochs=2,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                lora_rank=64,
                lora_alpha=128,
                size_factor=28,
                max_pixels=1025000,
                freeze_vit=True,
            ),
            config=orign_config,
        )

        val_sft_buffer = ReplayBuffer(
            name=val_adapter,
            vram_request="40Gi",
            dtype="bfloat16",
            train_every=50,
            sample_n=100,
            sample_strategy="Random",
            ms_swift_params=V1MSSwiftBufferParams(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                model_type="qwen2_5_vl",
                train_type="lora",
                deepspeed="zero3",
                torch_dtype="bfloat16",
                max_length=16384,
                val_split_ratio=1.0,
                num_train_epochs=2,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                lora_rank=64,
                lora_alpha=128,
                size_factor=28,
                max_pixels=1025000,
                freeze_vit=True,
            ),
            config=orign_config,
        )

        print("\n----\nchecking task: ", task.id)

        if "foo/train/status" in task.labels:
            console.print("task already trained", style="white")
            raise ValueError("Task already trained")

        if not task.episode:
            console.print("no episode", style="red")
            raise ValueError("Task has no episode")

        for i, action in enumerate(task.episode.actions):
            console.print("action: ", action, "\n\n")

            approved = False
            action_correction = None
            for review in action.reviews:
                if review.approved and review.reviewer_type == "human":
                    approved = True
                    action_correction = review.correction
                    break
            console.print("approved: ", approved)
            console.print("action_correction: ", action_correction)

            if "foo/train/status" in action.metadata:
                console.print("skipping train", style="white")
                continue

            if not action.prompt:
                console.print("no prompt", style="white")
                continue

            if not isinstance(action.prompt, ChatResponse):
                console.print("not a chat response", style="white")
                continue

            prompt: V1ChatEvent = action.prompt  # type: ignore
            if not prompt.request.prompt:
                console.print("no prompt", style="white")
                continue

            if not prompt.request.prompt.messages:
                console.print("no messages", style="red")
                continue

            if not prompt.request.prompt.messages[0].content or not isinstance(
                prompt.request.prompt.messages[0].content, list
            ):
                console.print("no content", style="red")
                continue

            if not prompt.request.prompt.messages[0].content[0].text:
                console.print("no text", style="red")
                continue

            if not prompt.request.prompt.messages[0].content[1].image_url:
                console.print("no image url", style="red")
                continue

            content = prompt.request.prompt.messages[0].content[0].text
            image_url = prompt.request.prompt.messages[0].content[1].image_url

            reason = None
            reason_update = None

            validation = None
            validation_update = None
            for reviewable in action.reviewables:
                if isinstance(reviewable, AnnotationReviewable):
                    if reviewable.key == "reason":
                        reason = reviewable.value
                        for review in reviewable.reviews:
                            if review.correction and review.reviewer_type == "human":
                                reason_update = review.correction
                                break
                    elif reviewable.key == "validation":
                        validation = reviewable.value
                        for review in reviewable.reviews:
                            if review.correction and review.reviewer_type == "human":
                                validation_update = review.correction
                                break

            console.print("reason: ", reason)
            console.print("validation: ", validation)

            console.print("reason_update: ", reason_update)
            console.print("validation_update: ", validation_update)

            if not reason:
                console.print("no reason", style="red")
                continue

            if approved:
                response_reason = reason
                if reason_update:
                    response_reason = reason_update

                response = f"<think>{response_reason}</think><answer>{action.action.model_dump_json()}</answer>"

                swift_prompt = {
                    "messages": [
                        {
                            "role": "user",
                            "content": content + " <image>",
                        },
                        {
                            "role": "assistant",
                            "content": response,
                        },
                    ],
                    "images": [image_url],
                }

                console.print("adding to actor buffer: ", swift_prompt)
                console.print("orignal prompt: ", prompt.model_dump())

                actor_sft_buffer.send([swift_prompt])

            if validation:
                val_ctx = (
                    "You are a helpful assistant judging if actions taken on a computer are correct. \n"
                    f"You are given a task to complete: '{task.description}'\n"
                    "You just took an action to complete that task. \n"
                    f"The action was: {action.action.model_dump_json()}\n"
                    "The first image I've provided you is the state of the computer before the action was taken, "
                    "and the second image is the state of the computer after the action was taken.\n"
                    "Using those images, you will decide if the action taken is correct.\n"
                    "Please return your decision in the format <think>...</think><answer>...</answer>\n"
                    "As an answer, please return 'yes' if the action was correct or 'no' if it was incorrect <image> <image>"
                )

                approved_text = "yes" if approved else "no"

                response_val = validation
                if validation_update:
                    response_val = validation_update

                response = (
                    f"<think>{response_val}</think><answer>{approved_text}</answer>"
                )

                if i + 1 < len(task.episode.actions):
                    next_action = task.episode.actions[i + 1]
                else:
                    console.print("no next action", style="red")
                    continue
                if not next_action.state.images:
                    console.print("no next state images", style="red")
                    continue

                end_state = next_action.state.images[0]

                val_swift_prompt = {
                    "messages": [
                        {"role": "user", "content": val_ctx},
                        {"role": "assistant", "content": response},
                    ],
                    "images": [image_url, end_state],
                }
                console.print("adding to val buffer: ", val_swift_prompt)

                val_sft_buffer.send([val_swift_prompt])

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on. Defaults to None.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """

        if not device:
            raise ValueError("This agent expects a desktop")

        # Post a message to the default thread to let the user know the task is in progress
        task.post_message("assistant", f"Starting task '{task.description}'")

        # Create threads in the task to update the user
        console.print("creating threads...")
        task.ensure_thread("debug")
        task.post_message("assistant", "I'll post debug messages here", thread="debug")

        # Check that the device we received is one we support
        if not isinstance(device, Desktop):
            raise ValueError("Only desktop devices supported")

        # Add standard agent utils to the device
        device.merge(AgentUtils())

        # Get the json schema for the tools
        tools = device.json_schema()
        console.print("tools: ", style="purple")
        console.print(JSON.from_data(tools))

        # Get info about the desktop
        info = device.info()
        screen_size = info["screen_size"]
        console.print(f"Screen size: {screen_size}")

        history: List[Step] = []

        actor = self.get_actor()

        # Loop to run actions
        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                step, done = self.take_action(device, task, actor, history)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

            if step:
                history.append(step)

            if done:
                console.print("task is done", style="green")
                # TODO: remove
                time.sleep(10)
                return task

            time.sleep(2)

        task.status = TaskStatus.FAILED
        task.save()
        task.post_message("assistant", "❗ Max steps reached without solving task")
        console.print("Reached max steps without solving task", style="red")

        return task

    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def take_action(
        self,
        device: Device,
        task: Task,
        actor: Actor,
        history: List[Step],
    ) -> Tuple[Optional[Step], bool]:
        """Take an action

        Args:
            desktop (Desktop): Desktop to use
            task (str): Task to accomplish
            actor (Actor): Actor to use
            history (List[Step]): History of steps taken

        Returns:
            Tuple[Optional[Step], bool]: A tuple containing the step taken and whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return None, True

            console.print("taking action...", style="white")

            step = actor.act(task, device, history)

            if step.task:
                task = step.task

            # The agent will return 'end' if it believes it's finished
            if step.action.name == "end":
                console.print("final result: ", style="green")
                console.print(JSON.from_data(step.action.parameters))
                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {step.action.parameters['value']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                return step, True

            # Find the selected action in the tool
            action = device.find_action(step.action.name)
            console.print(f"found action: {action}", style="blue")
            if not action:
                console.print(f"action returned not found: {step.action.name}")
                raise SystemError("action not found")

            # Take the selected action
            try:
                action_response = device.use(action, **step.action.parameters)
            except Exception as e:
                raise ValueError(f"Trouble using action: {e}")

            console.print(f"action output: {action_response}", style="blue")
            if action_response:
                task.post_message(
                    "assistant", f"👁️ Result from taking action: {action_response}"
                )

            # Record the action for feedback and tuning
            task.record_action(
                state=step.state,
                prompt=step.prompt,
                action=step.action,
                tool=device.ref(),
                result=action_response,
                agent_id=self.name(),
                model=step.model_id,
            )

            return step, False

        except Exception as e:
            print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def get_actor(self) -> Actor:
        return OrignActor()

    def label_task(
        self, remote: str, token: str, task: Task, key: str, value: str
    ) -> None:
        """Label a task as trained

        Args:
            task (Task): The task
        """
        update = V1TaskUpdate(
            set_labels={key: value},
        )
        resp = requests.put(
            f"{remote}/v1/tasks/{task.id}",
            json=update.model_dump(),
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[FooConfig]:
        """Type of config

        Returns:
            Type[DinoConfig]: Config type
        """
        return FooConfig

    @classmethod
    def from_config(cls, config: FooConfig) -> "Foo":
        """Create an agent from a config

        Args:
            config (DinoConfig): Agent config

        Returns:
            Foo: The agent
        """
        return Foo()

    @classmethod
    def default(cls) -> "Foo":
        """Create a default agent

        Returns:
            Foo: The agent
        """
        return Foo()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        # <INITIALIZE AGENT HERE>
        return


Agent = Foo
