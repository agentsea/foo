import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type

from agentdesk import Desktop
from devicebay import Device
from json_repair import repair_json
from mllm import Prompt as SkillPrompt
from orign import ChatModel
from orign.models import ChatResponse, Prompt
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState
from skillpacks.action_opts import ActionOpt
from skillpacks.server.models import V1Action
from surfkit.agent import TaskAgent
from taskara import Task, TaskStatus
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
)
from threadmem import RoleMessage, RoleThread
from toolfuse.util import AgentUtils

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)

WORKFLOW_MODEL_ID = "pbarker/Qwen2-VL-7B-sk-airbnb-full-10tsk-237ex-5epoch"


class FooConfig(BaseModel):
    pass


class Foo(TaskAgent):
    """A desktop agent that learns"""

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

        self.workflow_model_id = WORKFLOW_MODEL_ID
        self.workflow_model = ChatModel(model=self.workflow_model_id, provider="vllm")
        self.workflow_model.connect()

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

        thread: Optional[RoleThread] = None
        action: Optional[V1Action] = None
        state: Optional[EnvState] = None

        # Loop to run actions
        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                thread, done, action, state = self.take_action(
                    device, task, thread, action, state
                )
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

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
        desktop: Desktop,
        task: Task,
        thread: Optional[RoleThread] = None,
        previous_action: Optional[V1Action] = None,
        previous_state: Optional[EnvState] = None,
    ) -> Tuple[Optional[RoleThread], bool, Optional[V1Action], Optional[EnvState]]:
        """Take an action

        Args:
            desktop (Desktop): Desktop to use
            task (str): Task to accomplish
            thread (Optional[RoleThread]): Role thread for the task

        Returns:
            bool: Whether the task is complete
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
                return thread, True, None, None

            console.print("taking action...", style="white")

            _thread = RoleThread()

            # Take a screenshot of the desktop and post a message with it
            screenshots = desktop.take_screenshots(count=2)
            s0 = screenshots[0]
            width, height = s0.size  # Get the dimensions of the screenshot
            console.print(f"Screenshot dimensions: {width} x {height}")

            task.post_message(
                "assistant",
                "current image",
                images=screenshots,
                thread="debug",
            )

            # Get the current mouse coordinates
            x, y = desktop.mouse_coordinates()
            console.print(f"mouse coordinates: ({x}, {y})", style="white")

            ctx = (
                "You are a helpful assistant operating a computer. "
                f"You are given a task to complete: '{task.description}' "
                f"You have the following tools at your disposal: {desktop.json_schema()} "
                "I am going to provide you with screenshots of the current state of the computer, as well as "
                "a screenshot of the previous state of the computer, with the action that was taken to get to the current state. "
                "Using those screenshots, you will decide what action to take next. "
            )

            images = []
            if thread and previous_action and previous_state:
                ctx += f"\n\nThe previous state of the computer was <image> and the previous action was {previous_action.model_dump_json()}"
                images.append(previous_state.images[0])  # type: ignore

            images.extend(screenshots)

            image_str = "".join(["<image>" for _ in screenshots])
            ctx += f"\n\nThe current screenshots for the desktops are {image_str}"

            ctx += "\n\nPlease return the action you want to take as a raw JSON object."

            console.print("context: ", style="white")
            console.print(ctx, style="white")

            # Craft the message asking the MLLM for an action
            msg = RoleMessage(
                role="user",
                text=ctx,
                images=images,
            )
            _thread.add_msg(msg)

            # Make the action selection
            response = self.workflow_model.chat(
                prompt=Prompt(messages=_thread.to_orign().messages),
                # sampling_params=SamplingParams(
                #     n=4,
                # ),
            )
            if not isinstance(response, ChatResponse):
                raise ValueError(f"Expected a ChatResponse, got: {type(response)}")

            try:
                actions = self._parse_response(response)
                selection = self._select_action(actions)
                console.print("action selection: ", style="white")
                console.print(JSON.from_data(selection.model_dump()))

                task.post_message(
                    "assistant",
                    f"▶️ Taking action '{selection.name}' with parameters: {selection.parameters}",
                )

            except Exception as e:
                console.print(f"Response failed to parse: {e}", style="red")
                raise

            # The agent will return 'result' if it believes it's finished
            if selection.name == "result":
                console.print("final result: ", style="green")
                console.print(JSON.from_data(selection.parameters))
                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {selection.parameters['value']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                return _thread, True, None, None

            # Find the selected action in the tool
            action = desktop.find_action(selection.name)
            console.print(f"found action: {action}", style="blue")
            if not action:
                console.print(f"action returned not found: {selection.name}")
                raise SystemError("action not found")

            # Take the selected action
            try:
                action_response = desktop.use(action, **selection.parameters)
            except Exception as e:
                raise ValueError(f"Trouble using action: {e}")

            console.print(f"action output: {action_response}", style="blue")
            if action_response:
                task.post_message(
                    "assistant", f"👁️ Result from taking action: {action_response}"
                )

            response = RoleMessage(role="assistant", text=selection.model_dump_json())

            prompt = SkillPrompt(
                thread=_thread, response=response, model=self.workflow_model_id
            )
            state = EnvState(images=screenshots)

            # Record the action for feedback and tuning
            task.record_action(
                state=state,
                prompt=prompt,
                action=selection,
                tool=desktop.ref(),
                result=action_response,
                agent_id=self.name(),
                model=self.workflow_model_id,
                action_opts=[
                    ActionOpt(action=action, prompt=prompt) for action in actions
                ],
            )

            _thread.add_msg(response)
            return _thread, False, selection, state

        except Exception as e:
            print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def _parse_response(self, response: ChatResponse) -> List[V1Action]:
        output = []
        for choice in response.choices:
            try:
                obj = repair_json(choice.text, return_objects=True)
                action = V1Action.model_validate(obj)
                output.append(action)
            except Exception as e:
                print("Error parsing action: ", e)
                continue
        return output

    def _select_action(self, actions: List[V1Action]) -> V1Action:
        return actions[0]

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
