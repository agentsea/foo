from typing import List, Optional

from agentdesk import Desktop
from json_repair import repair_json
from mllm import Prompt as SkillPrompt
from orign import ChatModel
from orign.models import ChatResponse, Prompt, SamplingParams
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from taskara import Task
from threadmem import RoleMessage, RoleThread

from .base import Actor, Step

console = Console(force_terminal=True)


class SwiftActor(Actor[Desktop]):
    """An actor that uses ms-swift and orign"""

    def __init__(self, model: Optional[str] = None):
        self.workflow_model_id = model or "pbarker/Airbnb-CB0.1-50tsk-3epoch"
        self.workflow_model = ChatModel(model=self.workflow_model_id, provider="vllm")
        self.workflow_model.connect()

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        thread = RoleThread()

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
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
        x, y = device.mouse_coordinates()
        console.print(f"mouse coordinates: ({x}, {y})", style="white")

        ctx = (
            "You are a helpful assistant operating a computer. \n"
            f"You are given a task to complete: '{task.description}'\n"
            f"You have the following tools at your disposal: {device.json_schema()}\n\n"
            "I am going to provide you with screenshots of the current state of the computer, "
            "as well as\na screenshot of the previous state of the computer, with the action that "
            "was taken to get to the current state.\nUsing those screenshots, you will decide what "
            "action to take next.\n\nPlease return the action you want to take as a raw JSON object.\n\n"
            "The current screenshots for the desktops are <image>"
        )

        images = []

        images.extend(screenshots)

        console.print("context: ", style="white")
        console.print(ctx, style="white")

        # Craft the message asking the MLLM for an action
        msg = RoleMessage(
            role="user",
            text=ctx,
            images=images,
        )
        thread.add_msg(msg)

        # Make the action selection
        response = self.workflow_model.chat(
            prompt=Prompt(messages=thread.to_orign().messages),
            sampling_params=SamplingParams(
                n=4,
            ),
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

        response = RoleMessage(role="assistant", text=selection.model_dump_json())
        prompt = SkillPrompt(
            thread=thread, response=response, model=self.workflow_model_id
        )

        step = Step(
            state=EnvState(images=screenshots),
            action=selection,
            task=task,
            thread=thread,
            model_id=self.workflow_model_id,
            prompt=prompt,
        )

        return step

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
        console.print("action options: ", style="white")

        for i, act in enumerate(actions):
            console.print(f"Option {i + 1}:", style="yellow")
            console.print(JSON.from_data(act.model_dump()), style="blue")

        action = actions[0]
        # Remove start_x and start_y if present
        if action.parameters and isinstance(action.parameters, dict):  # type: ignore
            action.parameters.pop("start_x", None)
            action.parameters.pop("start_y", None)
        return action
