# type: ignore

import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type

import requests
from agentcore.models import V1UserProfile
from agentdesk import Desktop
from devicebay import Device
from nebu import V1EnvVar
from orign.config import GlobalConfig, ServerConfig
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks.reviewable import AnnotationReviewable, ReviewerType
from surfkit.agent import TaskAgent
from surfkit.auth.util import get_user_info
from surfkit.skill import Skill
from taskara import Task, TaskStatus
from taskara.server.models import V1TaskUpdate
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
)
from toolfuse.util import AgentUtils

from .actor_genx import Actor
from .actor_utils import Step, create_actor_prompt_for_sft
from .buffer import (
    create_actor_sft_buffer,
    # create_description_annot_sft_buffer,
    # create_reason_annot_sft_buffer,
    create_val_sft_buffer,
    # create_validation_annot_sft_buffer,
)
from .prompts import (
    # create_actor_prompt,
    create_description_prompt,
    # create_reason_actor_prompt,
    create_reason_prompt,
    create_validation_actor_prompt,
    create_validation_prompt,
    create_validation_reasoning_prompt,
)

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
        user: V1UserProfile,
    ):
        """Learn a task

        Args:
            task (Task): The task
            skill (Skill): The associated skill
        """
        print("\nlearning task: ", task.id, flush=True)
        print("with user: ", user.model_dump(), flush=True)
        from nebu import Namespace
        from orign.zoo.processors.unsloth_trainer import TrainingRequest, UnslothSFT

        if not task.remote or not task.auth_token:
            raise ValueError("Task remote or token not set")

        self._get_and_print_user_profile(
            task.auth_token, "task.auth_token from learn_task"
        )

        print("task remote: ", task.remote, flush=True)
        print("task auth_token: ", task.auth_token, flush=True)

        internal_auth_token = os.getenv("AGENTSEA_API_KEY")
        if not internal_auth_token:
            raise ValueError("AGENTSEA_API_KEY not set")

        self._get_and_print_user_profile(
            internal_auth_token, "internal_auth_token from learn_task"
        )

        if not skill.name:
            raise ValueError("Skill name not set")

        print(
            "\ncreating namespace with internal auth token: ",
            internal_auth_token,
            flush=True,
        )
        Namespace("agentsea", owner="agentsea", api_key=internal_auth_token)
        print("namespace created", flush=True)

        print("current env: ", os.environ)

        # server_config = ServerConfig(
        #     name="foo",
        #     api_key=internal_auth_token,
        #     server=os.getenv("ORIGN_SERVER"),
        #     auth_server=os.getenv("AGENTSEA_AUTH_SERVER"),
        # )
        # admin_orign_config = GlobalConfig(servers=[server_config], current_server="foo")

        user_server_config = ServerConfig(
            name="foo",
            api_key=task.auth_token,
            server=os.getenv("ORIGN_SERVER"),
            auth_server=os.getenv("AGENTSEA_AUTH_SERVER"),
        )
        user_orign_config = GlobalConfig(
            servers=[user_server_config], current_server="foo"
        )

        print("\nuser_orign_config: ", user_orign_config)
        print("user_server_config: ", user_server_config)

        if not task.owner_id:
            raise ValueError("Task owner_id not set")

        if not user:
            raise ValueError("User not set")

        actor_adapter = self.get_actor_adapter_name(skill, task.owner_id, user)
        val_adapter = self.get_val_adapter_name(skill, task.owner_id, user)

        print("\nactor_adapter: ", actor_adapter)
        print("val_adapter: ", val_adapter)

        if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            raise ValueError("HUGGINGFACE_HUB_TOKEN not set")

        print("creating unsloth processor...")
        train_unsloth_sft = UnslothSFT(
            namespace="agentsea",
            hot_reload=False,
            debug=True,
            env=[
                V1EnvVar(
                    key="HUGGINGFACE_HUB_TOKEN",
                    value=os.getenv("HUGGINGFACE_HUB_TOKEN"),
                ),
            ],
        )
        print("unsloth processor created: ", train_unsloth_sft)

        ###
        # Create all the buffers!
        ###
        actor_sft_buffer = create_actor_sft_buffer(
            actor_adapter, skill.id, user_orign_config
        )

        val_sft_buffer = create_val_sft_buffer(val_adapter, skill.id, user_orign_config)

        # reason_annot_sft_buffer = create_reason_annot_sft_buffer(
        #     skill.id, admin_orign_config
        # )

        # validation_annot_sft_buffer = create_validation_annot_sft_buffer(
        #     skill.id, admin_orign_config
        # )

        # description_annot_sft_buffer = create_description_annot_sft_buffer(
        #     skill.id, admin_orign_config
        # )

        print("\n----\nchecking task: ", task.id)

        if "foo/train/status" in task.labels:
            console.print("task already trained", style="white")
            raise ValueError("Task already trained")

        if not task.episode:
            console.print("no episode", style="red")
            raise ValueError("Task has no episode")

        send_val_sft = []
        send_actor_sft = []
        send_reason_annot_sft = []
        send_validation_annot_sft = []
        send_description_annot_sft = []

        # console.print("getting device...")
        # device = Desktop(
        #     agentd_url="http://localhost:8000", check_health=False, requires_proxy=False
        # )

        # console.print("getting ctx...")
        # content = Actor.get_ctx(task, device, [])
        # reason_content = Actor.get_reason_ctx(task, device, [])
        # console.print("got ctx")

        for i, action in enumerate(task.episode.actions):
            console.print("\n\n========\naction: ", action.__dict__, "\n\n")

            approved = False
            action_correction = None
            for review in action.reviews:
                if review.approved and review.reviewer_type in [
                    "human",
                    "user",
                ]:
                    approved = True
                    if isinstance(review.correction, dict):
                        action_correction = review.correction["value"]  # type: ignore
                    else:
                        action_correction = review.correction
                    break

            console.print("approved: ", approved)
            console.print("action_correction: ", action_correction)

            if not approved:
                console.print("skipping not approved", style="white")
                continue

            if "foo/train/status" in action.metadata:
                console.print("skipping train", style="white")
                continue

            if not action.state.images:
                console.print("no images", style="red")
                continue

            before_state = action.state.images[-1]

            reason = None
            reason_update = None
            reason_best = None

            validation = None
            validation_update = None
            validation_best = None

            description = None
            description_update = None
            description_best = None

            for reviewable in action.reviewables:
                console.print("\nreviewable: ", reviewable.to_v1().model_dump())
                if isinstance(reviewable, AnnotationReviewable):
                    console.print("\nreviewable: ", reviewable.to_v1().model_dump())
                    if reviewable.key == "reason":
                        reason = reviewable.value
                        reason_best = reason
                        for review in reviewable.reviews:
                            console.print(
                                "\nreason review: ", review.to_v1().model_dump()
                            )
                            if review.correction and review.reviewer_type in [
                                "human",
                                "user",
                            ]:
                                console.print(
                                    "\nreason correction: ", review.correction
                                )
                                if isinstance(review.correction, dict):
                                    reason_update = review.correction["value"]  # type: ignore
                                    reason_best = reason_update
                                else:
                                    reason_update = review.correction
                                    reason_best = reason_update
                                break
                    elif reviewable.key == "validation":
                        validation = reviewable.value
                        validation_best = validation
                        for review in reviewable.reviews:
                            console.print(
                                "\nvalidation review: ", review.to_v1().model_dump()
                            )
                            if review.correction and review.reviewer_type in [
                                "human",
                                "user",
                            ]:
                                console.print(
                                    "\nvalidation correction: ", review.correction
                                )
                                if isinstance(review.correction, dict):
                                    validation_update = review.correction["value"]  # type: ignore
                                    validation_best = validation_update
                                else:
                                    validation_update = review.correction
                                    validation_best = validation_update
                                break
                    elif reviewable.key == "description":
                        description = reviewable.value
                        description_best = description
                        for review in reviewable.reviews:
                            console.print(
                                "\ndescription review: ", review.to_v1().model_dump()
                            )
                            if review.correction and review.reviewer_type in [
                                "human",
                                "user",
                            ]:
                                console.print(
                                    "\ndescription correction: ", review.correction
                                )
                                if isinstance(review.correction, dict):
                                    description_update = review.correction["value"]  # type: ignore
                                    description_best = description_update
                                else:
                                    description_update = review.correction
                                    description_best = description_update
                                break

            console.print("\nreason: ", reason)
            console.print("\nvalidation: ", validation)
            console.print("\ndescription: ", description)

            console.print("\nreason_update: ", reason_update)
            console.print("\nvalidation_update: ", validation_update)
            console.print("\ndescription_update: ", description_update)

            console.print("\nreason_best: ", reason_best)
            console.print("\nvalidation_best: ", validation_best)
            console.print("\ndescription_best: ", description_best)

            if not reason:
                console.print("no reason", style="red")
                continue

            response_reason = reason
            if reason_update:
                response_reason = reason_update

            if reason_update:
                console.print(
                    "adding to reason annot dpo buffer: ", response_reason, reason
                )
                # oai_reason_dpo_prompt = {  # type: ignore
                #     "messages": [
                #         {
                #             "role": "user",
                #             "content": reason_content + " <image>",
                #         },
                #         {
                #             "role": "assistant",
                #             "content": response_reason,
                #         },
                #     ],
                #     "images": [before_state],
                #     "rejected_response": reason,
                # }

                # send_actor_dpo.append(oai_reason_dpo_prompt)
                # send_base_actor_dpo.append(oai_reason_dpo_prompt)

            action_str = action.action.model_dump_json(
                exclude_unset=True, exclude_none=True, exclude_defaults=True
            )

            has_next_action = False
            next_action = None
            if i + 1 < len(task.episode.actions):
                next_action = task.episode.actions[i + 1]
                has_next_action = True
            else:
                console.print("no next action", style="red")
            if not next_action or not next_action.state.images:
                console.print("no next state images", style="red")
                has_next_action = False

            end_state = None
            if has_next_action:
                end_state = next_action.state.images[0]  # type: ignore

            console.print("end_state: ", end_state)
            console.print("has_next_action: ", has_next_action)
            console.print("next_action: ", next_action)

            if not task.description:
                raise ValueError("Task description not set")

            if reason_best and end_state:
                reason_oai_prompt = create_reason_prompt(
                    image1_url=before_state,
                    image2_url=end_state,
                    action=action_str,
                    task_description=task.description,
                    answer=reason_best,
                )

                if approved:
                    console.print("adding to reason annot buffer: ", reason_oai_prompt)
                    send_reason_annot_sft.append(reason_oai_prompt)

                if reason_update:
                    console.print(
                        f"adding to reason annot dpo buffer \n-- good reason: {reason_best}\n bad reason: {reason}\n ",
                    )
                    reason_oai_prompt_copy = reason_oai_prompt.copy()
                    reason_oai_prompt_copy["rejected_response"] = reason
                    # send_reason_annot_dpo.append(reason_oai_prompt_copy)

            if description_best and end_state:
                description_oai_prompt = create_description_prompt(
                    image1_url=before_state,
                    image2_url=end_state,
                    action=action_str,
                    answer=description_best,
                )

                if approved:
                    console.print(
                        "adding to description annot buffer: ", description_oai_prompt
                    )
                    send_description_annot_sft.append(description_oai_prompt)

                if description_update:
                    console.print(
                        f"adding to description annot dpo buffer\n-- good description: {description_best}\n bad description: {description}\n ",
                    )
                    description_oai_prompt_copy = description_oai_prompt.copy()
                    description_oai_prompt_copy["rejected_response"] = description
                    # send_description_annot_dpo.append(description_oai_prompt_copy)

            if validation_best and end_state:
                validation_oai_prompt = create_validation_prompt(
                    image1_url=before_state,
                    image2_url=end_state,
                    action=action_str,
                    task_description=task.description,
                    answer=validation_best,
                )

                if approved:
                    console.print(
                        "adding to validation annot buffer: ", validation_oai_prompt
                    )
                    send_validation_annot_sft.append(validation_oai_prompt)

                if validation_update:
                    console.print(
                        f"adding to validation annot dpo buffer \n-- good validation: {validation_best}\n bad validation: {validation}\n ",
                    )
                    validation_oai_prompt_copy = validation_oai_prompt.copy()
                    validation_oai_prompt_copy["rejected_response"] = validation
                    # send_validation_annot_dpo.append(validation_oai_prompt_copy)

            if approved:
                oai_prompt = create_actor_prompt_for_sft(
                    task=task,
                    reason=response_reason,
                    action=action.action,
                    image_url=before_state,
                )

                # response = (
                #     f"<think>{response_reason}</think><answer>{action_str}</answer>"
                # )

                # oai_reason_prompt = create_reason_actor_prompt(
                #     content=reason_content,
                #     image_url=before_state,
                #     response=response_reason,
                # )

                send_actor_sft.append(oai_prompt)
                # send_actor_sft.append(oai_reason_prompt)
                # send_base_actor_sft.append(oai_prompt)
                # send_base_actor_sft.append(oai_reason_prompt)

                console.print("adding to actor buffer: ", oai_prompt)
                # console.print("orignal prompt: ", prompt.model_dump())

                # {"messages": [{"role": "system", "content": "You are a useful and harmless math calculator"}, {"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}, {"role": "assistant", "content": "It equals 3"}], "rejected_response": "I don't know"}

            if validation and end_state:
                console.print("adding to val sft buffer...")
                val_ctx = (
                    "You are a helpful assistant judging if actions taken on a computer are correct. \n"
                    f"You are given a task to complete: '{task.description}'\n"
                    "You just took an action to complete that task. \n"
                    f"The action was: {action.action.model_dump_json()}\n"
                    "The first image I've provided you is the state of the computer before the action was taken, "
                    "and the second image is the state of the computer after the action was taken.\n"
                    "Using those images, you will decide if the action taken is correct.\n"
                    "Please return your decision in the format <think>...</think><answer>...</answer>\n"
                    "As an answer, please return 'yes' if the action was correct or 'no' if it was incorrect"
                )

                val_ctx_reason = (
                    "You are a helpful assistant judging if actions taken on a computer are correct. \n"
                    f"You are given a task to complete: '{task.description}'\n"
                    "You just took an action to complete that task. \n"
                    f"The action was: {action.action.model_dump_json()}\n"
                    "The first image I've provided you is the state of the computer before the action was taken, "
                    "and the second image is the state of the computer after the action was taken.\n"
                    "Using those images, you will reason about why the action was correct or incorrect.\n"
                    "Please return your reasoning as a plain text response."
                )

                approved_text = "yes" if approved else "no"

                response_val = validation
                if validation_update:
                    response_val = validation_update

                response = (
                    f"<think>{response_val}</think><answer>{approved_text}</answer>"
                )

                val_oai_prompt = create_validation_actor_prompt(
                    val_ctx=val_ctx,
                    image1_url=before_state,
                    image2_url=end_state,
                    response=response,
                )
                console.print("adding to val buffer: ", val_oai_prompt)
                send_val_sft.append(val_oai_prompt)

                val_ctx_reason_oai_prompt = create_validation_reasoning_prompt(
                    val_ctx_reason=val_ctx_reason,
                    image1_url=before_state,
                    image2_url=end_state,
                    response=response_val,
                )
                send_val_sft.append(val_ctx_reason_oai_prompt)

                if validation_update:
                    console.print(
                        "adding to val dpo buffer: ",
                        validation_best,
                        validation,
                    )
                    val_ctx_reason_oai_prompt_copy = val_ctx_reason_oai_prompt.copy()
                    val_ctx_reason_oai_prompt_copy["rejected_response"] = validation  # type: ignore
                    # send_val_dpo.append(val_ctx_reason_oai_prompt_copy)

        if send_val_sft:
            console.print("sending to val sft buffer...")
            val_sft_buffer.send(send_val_sft)
            dataset = val_sft_buffer.sample(n=50, link=True)
            if not dataset.dataset_uri:
                raise ValueError("Dataset URI not found")

            print("training validation buffer...")
            print("training with user auth token: ", task.auth_token)
            train_unsloth_sft(
                data=TrainingRequest(  # type: ignore
                    adapter=self.get_val_adapter_name(skill, task.owner_id, user),
                    dataset=dataset.dataset_uri,
                    labels={"skill": skill.id, "task": task.id},
                ),
                api_key=internal_auth_token,
                user_key=task.auth_token,
            )
            print("sent training to validation buffer")

        if send_actor_sft:
            console.print("sending to actor sft buffer...")
            actor_sft_buffer.send(send_actor_sft)

            dataset = actor_sft_buffer.sample(n=50, link=True)
            if not dataset.dataset_uri:
                raise ValueError("Dataset URI not found")

            print("training actor buffer...")
            print("training with user auth token: ", task.auth_token)
            train_unsloth_sft(
                data=TrainingRequest(  # type: ignore
                    adapter=self.get_actor_adapter_name(skill, task.owner_id, user),
                    dataset=dataset.dataset_uri,
                    labels={"skill": skill.id, "task": task.id},
                ),
                api_key=internal_auth_token,
                user_key=task.auth_token,
            )
            print("sent training to actor buffer")

        if send_reason_annot_sft:
            console.print("sending to reason annot sft buffer...")
            # reason_annot_sft_buffer.send(send_reason_annot_sft)

        if send_validation_annot_sft:
            console.print("sending to validation annot sft buffer...")
            # validation_annot_sft_buffer.send(send_validation_annot_sft)

        if send_description_annot_sft:
            console.print("sending to description annot sft buffer...")
            # description_annot_sft_buffer.send(send_description_annot_sft)

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
        from nebu import Namespace

        print("creating namespace...")
        Namespace("agentsea", owner="agentsea")
        print("namespace created")
        try:
            if not device:
                raise ValueError("This agent expects a desktop")

            # Post a message to the default thread to let the user know the task is in progress
            task.post_message(
                "assistant",
                "I'm a new agent. My last update was 2025-05-22, around noon EET.",
            )
            task.post_message("assistant", f"Starting task '{task.description}'")

            # Create threads in the task to update the user
            console.print("creating threads...")
            task.ensure_thread("debug")
            task.post_message(
                "assistant", "I'll post debug messages here", thread="debug"
            )

            if not task.auth_token:
                raise ValueError("Task auth token not set")

            self._get_and_print_user_profile(
                task.auth_token, "task.auth_token from solve_task"
            )

            internal_auth_token = os.getenv("AGENTSEA_API_KEY")
            if not internal_auth_token:
                raise ValueError("AGENTSEA_API_KEY not set")

            print("device: ", device)
            print("device type: ", type(device))
            print("device dict: ", device.__dict__)
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

            if not task.auth_token:
                raise ValueError("Task auth token not set")

            console.print("getting user info...")
            user = get_user_info(task.auth_token)
            console.print("got user info")

            skill = self.get_skill(task)

            if not task.owner_id:
                raise ValueError("Task owner_id not set")

            if not user:
                raise ValueError("User not set")

            adapter = self.get_actor_adapter_name(skill, task.owner_id, user)
            print("adapter: ", adapter)

            console.print("getting actor...")
            actor = self.get_actor(
                api_key=internal_auth_token,
                adapter=adapter,
                user_key=task.auth_token,
            )
            console.print("got actor")

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

                # time.sleep(2)

            task.status = TaskStatus.FAILED
            task.save()
            task.post_message("assistant", "❗ Max steps reached without solving task")
            console.print("Reached max steps without solving task", style="red")

            return task

        except Exception as e:
            console.print(f"Error: {e}", style="red")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.save()
            task.post_message("assistant", f"❗ Error solving task: {e}")
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
        start_time = time.time()
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

            refresh_time = time.time()
            console.print(
                f"Refresh took {refresh_time - start_time} seconds", style="white"
            )

            console.print("taking action...", style="white")
            action_start_time = time.time()

            step = actor.act(task, device, history)  # type: ignore
            action_end_time = time.time()
            console.print(
                f"Actor took {action_end_time - action_start_time} seconds",
                style="white",
            )

            reviewable = AnnotationReviewable(
                key="reason",
                value=step.reason,  # type: ignore
                annotator=self.name(),
                annotator_type=ReviewerType.AGENT.value,
            )

            record_action_start_time = time.time()
            # Record the action for feedback and tuning
            task.record_action(
                state=step.state,
                action=step.action,
                tool=device.ref(),
                result=step.result,
                agent_id=self.name(),
                model=step.model_id,
                reviewables=[reviewable],
            )
            record_action_end_time = time.time()
            console.print(
                f"Record action took {record_action_end_time - record_action_start_time} seconds",
                style="white",
            )

            end_time = time.time()
            console.print(
                f"Total took {end_time - start_time} seconds",
                style="white",
            )

            return step, step.end

        except Exception as e:
            print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def get_actor_adapter_name(
        self, skill: Skill, owner_id: str, user: V1UserProfile
    ) -> str:
        if "@" in owner_id:  # TODO: yuck, user owner_id in the adapter to specify
            namespace = user.handle  # type: ignore
        else:
            print("actor name using org_id: ", owner_id)
            if not user.organizations:
                raise ValueError("User has no organizations")

            if owner_id not in user.organizations.keys():
                raise ValueError(
                    f"User {user.handle} does not have access to organization {owner_id}"
                )
            print("org_info: ", user.organizations[owner_id])
            org_info = user.organizations[owner_id]
            namespace = org_info["org_name"]

        return f"{namespace}/{skill.name.lower().replace(' ', '-')}-act"  # type: ignore

    def get_val_adapter_name(
        self, skill: Skill, owner_id: str, user: V1UserProfile
    ) -> str:
        if "@" in owner_id:
            print("val name using user handle: ", user.handle)
            namespace = user.handle  # type: ignore
        else:
            if not user.organizations:
                raise ValueError("User has no organizations")

            if owner_id not in user.organizations.keys():
                raise ValueError(
                    f"User {user.handle} does not have access to organization {owner_id}"
                )
            print("val name using org_info: ", user.organizations[owner_id])
            org_info = user.organizations[owner_id]
            namespace = org_info["org_name"]

        return f"{namespace}/{skill.name.lower().replace(' ', '-')}-validate"  # type: ignore

    def get_skill(self, task: Task) -> Skill:
        skill_id = None
        if task.skill:
            skill_id = task.skill
        elif "skill" in task.labels:
            skill_id = task.labels["skill"]
        elif "skill_id" in task.labels:
            skill_id = task.labels["skill_id"]
        else:
            raise ValueError("Task skill or skill label not set")

        console.print(f"finding skill_id: {skill_id}")
        skills = Skill.find(id=skill_id, remote=task.remote, token=task.auth_token)
        if not skills:
            raise ValueError(f"Skill not found: {skill_id}")
        skill = skills[0]
        console.print(f"found skill: {skill.id}")

        return skill

    def get_actor(
        self,
        api_key: str,
        adapter: str,
        user_key: Optional[str] = None,
    ) -> Actor:
        return Actor(adapter_name=adapter, api_key=api_key, user_key=user_key)

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

    def _get_and_print_user_profile(self, token: str, token_description: str):
        if not token:
            print(
                f"Token '{token_description}' is not set. Skipping user profile fetch.",
                flush=True,
            )
            return

        auth_server_url = os.getenv("AGENTSEA_AUTH_SERVER")
        if not auth_server_url:
            print(
                "AGENTSEA_AUTH_SERVER environment variable is not set. Skipping user profile fetch.",
                flush=True,
            )
            return

        print(
            f"Attempting to get user profile with {token_description}: {token[:5]}...",
            flush=True,
        )
        try:
            response = requests.get(
                f"{auth_server_url}/v1/users/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            user_profile = response.json()
            print(
                f"User profile ({token_description}): ",
                JSON.from_data(user_profile),
                flush=True,
            )
        except requests.exceptions.RequestException as e:
            print(f"Error getting user profile ({token_description}): {e}", flush=True)
        except Exception as e:
            print(
                f"An unexpected error occurred while fetching user profile ({token_description}): {e}",
                flush=True,
            )


Agent = Foo
