from typing import List, Optional

from agentdesk import Desktop
from json_repair import repair_json
from orign import Adapter, ChatModel, V1ChatEvent
from orign.config import GlobalConfig
from orign.models import (
    ChatRequest,
    ChatResponse,
    ContentItem,
    ImageUrlContent,
    MessageItem,
    Prompt,
    SamplingParams,
)
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from taskara import Task

from .base import Actor, ReasonedAction, Step
from .img import image_to_b64

console = Console(force_terminal=True)


class OrignActor(Actor[Desktop]):
    """An actor that uses ms-swift and orign"""

    def __init__(
        self,
        api_key: str,
        adapter: Optional[str] = None,
        model: Optional[str] = None,
    ):
        config = GlobalConfig(api_key=api_key)

        self.adapter = None

        if adapter:
            console.print(f"getting adapter: {adapter}")
            adapters = Adapter.get(name=adapter, config=config)
            console.print(f"found adapters: {adapters}")
            if not adapters:
                console.print(
                    f"no adapters found for {adapter}, continuing without it..."
                )
                self.adapter = None
            else:
                self.adapter = adapter

        self.workflow_model_id = model or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.workflow_model = ChatModel(
            model=self.workflow_model_id,
            provider="vllm",
            config=config,
        )
        self.workflow_model.connect()

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        screenshot = image_to_b64(s0)

        # Get the current mouse coordinates
        x, y = device.mouse_coordinates()
        console.print(f"mouse coordinates: ({x}, {y})", style="white")

        ctx = self.get_ctx(task, device, history)

        console.print("context: ", style="white")
        console.print(ctx, style="white")

        content = [ContentItem(type="text", text=ctx)]
        content.append(
            ContentItem(type="image_url", image_url=ImageUrlContent(url=screenshot))
        )

        messages = [
            MessageItem(role="user", content=ctx),
        ]

        prompt = Prompt(messages=messages)
        sampling_params = SamplingParams(n=1)

        # Make the action selection
        response = self.workflow_model.chat(
            prompt=prompt,
            sampling_params=sampling_params,
            adapter=self.adapter,
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
                f"▶️ Taking action '{selection.action.name}' with parameters: {selection.action.parameters}",
            )

            console.print("reason: ", style="white")
            console.print(selection.reason, style="white")

        except Exception as e:
            console.print(f"Response failed to parse: {e}", style="red")
            raise

        event = V1ChatEvent(
            request=ChatRequest(prompt=prompt, sampling_params=sampling_params),
            response=response,
        )

        step = Step(
            state=EnvState(images=screenshots),
            action=selection.action,
            task=task,
            thread=messages,
            model_id=self.workflow_model_id,
            prompt=event,
            reason=selection.reason,
        )

        return step

    def _parse_response(self, response: ChatResponse) -> List[ReasonedAction]:
        import re

        output = []
        for choice in response.choices:
            text = choice.text

            console.print(f"choice text: {text}")

            # Extract the <think> ... </think> content (optional)
            think_match = re.search(r"<think>(.+?)</think>", text, re.DOTALL)

            if not think_match:
                console.print("Error: could not find <think> block in the response")
                continue

            thought_content = think_match.group(1).strip()
            console.print(f"parsed thought: {thought_content}")

            # Extract the <answer> ... </answer> content
            answer_match = re.search(r"<answer>(.+?)</answer>", text, re.DOTALL)
            if not answer_match:
                console.print("Error: could not find <answer> block in the response")
                continue

            answer_text = answer_match.group(1).strip()

            # Try to parse the JSON content from the <answer> block
            try:
                obj = repair_json(answer_text, return_objects=True)
                action = V1Action.model_validate(obj)
                output.append(ReasonedAction(action=action, reason=thought_content))
            except Exception as e:
                console.print("Error parsing action: ", e)
                continue

        if not output:
            raise ValueError("No valid actions found in the response")

        return output

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

    def get_ctx(self, task: Task, device: Desktop, history: List[Step]) -> str:
        return (
            "You are a helpful assistant operating a computer. \n"
            f"You are given a task to complete: '{task.description}'\n"
            f"You have the following tools at your disposal: {device.json_schema()}\n\n"
            "I am going to provide you with a screenshot of the current state of the computer, "
            "based on this image, you will decide what action to take next.\n"
            "Please return your response formatted like <think>...</think><answer>...</answer> "
            "where the answer is the action you want to take as a raw JSON object.\n\n"
            "I've provided you with the most recent screenshot of the desktop."
        )

    def get_reason_ctx(self, task: Task, device: Desktop, history: List[Step]) -> str:
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
