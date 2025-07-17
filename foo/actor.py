import os
import time

from typing import List, Optional

from agentdesk import Desktop
from chatmux.openai import (
    ChatRequest,
    ChatResponse,
)
from nebu import V1EnvVar
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState
from taskara import Task, TaskStatus

from .actor_utils import (
    ReasonedAction,
    Step,
    build_actor_messages_chatmux,
    parse_response,
)
from .img import upload_image_to_gcs

console = Console(force_terminal=True)


class V1ChatEvent(BaseModel):
    """A chat event"""

    request: ChatRequest
    response: ChatResponse


class Actor:
    """An actor that uses ms-swift and orign"""

    def __init__(self, adapter_name: str, api_key: str, user_key: Optional[str] = None):
        from orign import Adapter

        from .qwen_server import QwenVLServer

        if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            raise ValueError("HUGGINGFACE_HUB_TOKEN not set")

        env = [
            V1EnvVar(
                key="HUGGINGFACE_HUB_TOKEN",
                value=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            ),
        ]

        self.model = QwenVLServer(
            namespace="agentsea", hot_reload=False, debug=True, env=env
        )
        self.model.api_key = api_key  # type: ignore
        self.user_key = user_key  # type: ignore
        self.adapter_name = adapter_name

        namespace, name = adapter_name.split("/")
        console.print(f"checking for adapter: {namespace}/{name}", style="white")
        adapters = Adapter.get(namespace=namespace, name=name, api_key=user_key)
        if not adapters:
            console.print(
                f"\n>>>> Adapter {adapter_name} not found, using default", style="red"
            )
            self.adapter_name = "agentsea/Qwen2.5-VL-32B-Instruct-CARL-Gflights"

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        start_time = time.time()
        skill = task.skill
        if not skill:
            raise ValueError("No skill found")

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        gcs_url = upload_image_to_gcs(s0)
        print("gcs_url", gcs_url)

        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        # screenshot_b64 = image_to_b64(s0)
        screenshot_time = time.time()
        console.print(
            f"Screenshot took {screenshot_time - start_time} seconds", style="white"
        )

        # Get the current mouse coordinates
        x, y = device.mouse_coordinates()
        console.print(f"mouse coordinates: ({x}, {y})", style="white")

        # ctx = self.get_ctx(task, device, history)

        # console.print("context: ", style="white")
        # console.print(ctx, style="white")

        messages = build_actor_messages_chatmux(task, history, gcs_url)
        console.print(f"created {len(messages)} messages", style="white")

        # Create the full request object using the new ChatRequest model
        request = ChatRequest(  # type: ignore
            model=self.adapter_name,
            messages=messages,
            n=1,
            max_tokens=2048,
        )

        print("request", request)
        chat_start_time = time.time()
        response = self.model(request, poll=True, user_key=task.auth_token)  # type: ignore
        print("response", response)
        chat_end_time = time.time()
        console.print(
            f"Chat took {chat_end_time - chat_start_time} seconds", style="white"
        )

        if not isinstance(response, ChatResponse):
            # TODO: fix in processor
            if not response:
                raise ValueError("No response found")
            content = response.get("content", None)
            if not content:
                raise ValueError("No content found in response")
            response = ChatResponse.model_validate(content)

        raw_response = None
        try:
            raw_response = response.choices[0].message.content
        except Exception as e:
            console.print(f"Error parsing response: {e}", style="red")
            raise

        try:
            actions = parse_response(response)
            selection = self._select_action(actions)
            console.print("action selection: ", style="white")
            console.print(JSON.from_data(selection.model_dump()))

            task.post_message(
                "assistant",
                f"💭 {selection.reason} \n\n 📝 {selection.note} \n\n 🎯 {selection.description}",
            )
            task.post_message(
                "assistant",
                f"▶️ Taking action '{selection.action.name}' with parameters: {selection.action.parameters}",
            )

        except Exception as e:
            console.print(f"Response failed to parse: {e}", style="red")
            raise

        # The agent will return 'end' if it believes it's finished
        end = False
        if selection.action.name == "end" or selection.action.name == "result":
            console.print("final result: ", style="green")
            console.print(JSON.from_data(selection.action.parameters))
            result = (
                selection.action.parameters["result"]
                if "result" in selection.action.parameters
                else selection.action.parameters["value"]
                if "value" in selection.action.parameters
                else "I'm done!"
            )
            task.post_message(
                "assistant",
                f"✅ I think the task is done, please review the result: {result}",
            )
            task.status = TaskStatus.FINISHED
            task.save()
            end = True

        if selection.action.name == "use_secret":
            selection.action.parameters["secret_server"] = os.getenv("SERVER_ADDRESS")

        # Find the selected action in the tool
        action = device.find_action(selection.action.name)
        console.print(f"found action: {action}", style="blue")
        if not action:
            console.print(f"action returned not found: {selection.action.name}")
            raise SystemError("action not found")

        # Take the selected action
        try:
            action_start_time = time.time()
            action_response = device.use(action, **selection.action.parameters)
            action_end_time = time.time()
            console.print(
                f"Action took {action_end_time - action_start_time} seconds",
                style="white",
            )
        except Exception as e:
            raise ValueError(f"Trouble using action: {e}")

        console.print(f"action output: {action_response}", style="blue")
        if action_response:
            task.post_message(
                "assistant", f"👁️ Result from taking action: {action_response}"
            )

        if not selection.reason:
            selection.reason = "I will do the following."

        event = V1ChatEvent(
            request=request,
            response=response,
        )

        step = Step(
            state=EnvState(images=screenshots),
            action=selection.action,
            task=task,
            thread=messages,  # type: ignore
            model_id=self.adapter_name,
            prompt=event, # type: ignore
            reason=selection.reason,
            end=end,
            result=action_response,
            raw_response=raw_response,
            note=selection.note,
            description=selection.description,
            thought=selection.reason,
        )
        step_end_time = time.time()
        console.print(f"Step took {step_end_time - start_time} seconds", style="white")

        return step

    def _select_action(self, actions: List[ReasonedAction]) -> ReasonedAction:
        console.print("action options: ", style="white")

        for i, act in enumerate(actions):
            console.print(f"Option {i + 1}:", style="yellow")
            console.print(JSON.from_data(act.model_dump()), style="blue")

        action = actions[0]
        # Remove start_x and start_y if present
        if action.action.parameters and isinstance(action.action.parameters, dict):  # type: ignore
            action.action.parameters.pop("start_x", None)
            action.action.parameters.pop("start_y", None)
        return action

    @classmethod
    def get_ctx(cls, task: Task, device: Desktop, history: List[Step]) -> str:
        return (
            "You are a helpful assistant operating a computer. \n"
            f"You are given a task to complete: '{task.description}'\n"
            f"You have the following tools at your disposal: {device.json_schema()}\n\n"
            "I am going to provide you with a screenshot of the current state of the computer, "
            "based on this image, you will decide what action to take next.\n"
            "Please return your response formatted like <think>...</think><answer>...</answer> "
            "where the answer is the action you want to take as a raw JSON object.\n\n"
            "When you finish the task, please return the action 'end'.\n"
            "I've provided you with the most recent screenshot of the desktop."
        )

    @classmethod
    def get_reason_ctx(cls, task: Task, device: Desktop, history: List[Step]) -> str:
        return (
            "You are a helpful assistant operating a computer. \n"
            f"You are given a task to complete: '{task.description}'\n"
            f"You have the following tools at your disposal: {device.json_schema()}\n\n"
            "I am going to provide you with a screenshot of the current state of the computer, "
            "based on this image, you will reason about what you should do next to complete the task.\n"
            "Please return your reasoning as a plain text response."
            "For example, if the task is 'navigate to airbnb.com' and the screenshot shows a desktop you may reason: "
            "'I need to open the browser to navigate to airbnb.com, to do that I need to move the cursor over the browser icon. It isn't over that now so I need to move it there.'"
        )
