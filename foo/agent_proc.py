import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

from agentdesk import Desktop
from chatmux.openai import (
    ChatRequest,
    ChatResponse,
    ImageContentPart,
    ImageUrl,
    RequestMessage,
    TextContentPart,
    UserMessage,
    UserMessageContent,
    UserMessageContentPart,
)
from devicebay import Device
from json_repair import repair_json
from orign import Message, processor
from orign.config import GlobalConfig
from PIL import Image
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from taskara import Task, TaskStatus, V1Task

console = Console(force_terminal=True)


class V1ChatEvent(BaseModel):
    """A chat event"""

    request: ChatRequest
    response: ChatResponse


@dataclass
class Step:
    """A step in an episode"""

    state: EnvState
    action: V1Action
    action_opts: Optional[List[V1Action]] = None
    thread: Optional[List[RequestMessage]] = None
    task: Optional[Task] = None
    model_id: Optional[str] = None
    prompt: Optional[V1ChatEvent] = None
    reason: Optional[str] = None


class ReasonedAction(BaseModel):
    action: V1Action
    reason: str


def act(task: Task, device: Desktop, history: List[Step]) -> Step:
    skill = task.skill
    if not skill:
        raise ValueError("No skill found")

    actor = f"{skill}-actor"

    # Take a screenshot of the desktop and post a message with it
    screenshots = device.take_screenshots(count=1)
    s0 = screenshots[0]
    width, height = s0.size  # Get the dimensions of the screenshot
    console.print(f"Screenshot dimensions: {width} x {height}")

    screenshot_b64 = image_to_b64(s0)
    screenshot_uri = f"data:image/png;base64,{screenshot_b64}"  # Format as data URI

    # Get the current mouse coordinates
    x, y = device.mouse_coordinates()
    console.print(f"mouse coordinates: ({x}, {y})", style="white")

    ctx = get_ctx(task, device, history)

    console.print("context: ", style="white")
    console.print(ctx, style="white")

    # Construct messages using new models
    messages = [
        UserMessage(
            role="user",
            name=None,
            content=UserMessageContent(
                root=[
                    UserMessageContentPart(root=TextContentPart(type="text", text=ctx)),
                    UserMessageContentPart(
                        root=ImageContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=screenshot_uri, detail="auto"),
                        )
                    ),
                ]
            ),
        )
    ]

    # Create the full request object using the new ChatRequest model
    request = ChatRequest(  # type: ignore
        model=actor,
        messages=messages,
        n=1,
    )

    # Make the action selection
    # Pass the full request object to the chat method
    # Assuming self.workflow_model.chat now accepts ChatRequest and returns ChatResponse
    response = self.workflow_model.chat(
        request=request,
        adapter=self.adapter,
    )
    # Update type check to use the new ChatResponse model
    if not isinstance(response, ChatResponse):
        raise ValueError(f"Expected a ChatResponse, got: {type(response)}")

    try:
        actions = parse_response(response)
        selection = select_action(actions)
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
        request=request,
        response=response,
    )

    step = Step(
        state=EnvState(images=screenshots),
        action=selection.action,
        task=task,
        thread=messages,  # type: ignore
        model_id=actor,
        prompt=event,  # Keep event containing request/response? Linter warning needs check
        reason=selection.reason,
    )

    return step


setup = """
pip install surfkit chatmux orign rich
"""


@processor(image="python:3.11-slim", platform="runpod", setup_script=setup)
def agent(message: Message[V1Task]) -> V1Task:
    print(message)

    v1task = message.content
    if not v1task:
        raise ValueError("No task found")

    task = Task.from_v1(v1task)
    if not task.skill:
        raise ValueError("No skill found")

    api_key = task.auth_token
    if api_key is None:
        print("No Api key/token on Task or in Auth")

    try:
        config = Device.connect_config_type()(
            **{**task.device.config, "api_key": api_key}  # type: ignore
        )
        device = Device.connect(config=config)
    except Exception as e:
        err = f"error connecting to device: {e}"
        task.error = err
        task.status = TaskStatus.ERROR
        task.save()
        raise Exception(err)

    history = []

    for i in range(task.max_steps):
        print(f"Step {i+1} of {task.max_steps}")

        step = act(task, device, history)
        history.append(step)

        if task.is_done():
            task.status = TaskStatus.FINISHED
            task.save()
            print("Task done")
            break

    return v1task


def parse_response(response: ChatResponse) -> List[ReasonedAction]:
    import re

    output = []
    for choice in response.choices:
        # Access content from the message object within the choice using new structure
        text = choice.message.content if choice.message else None

        if not text:
            console.print("Warning: Choice message content is empty.")
            continue

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


def get_ctx(task: Task, device: Desktop, history: List[Step]) -> str:
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


def get_reason_ctx(task: Task, device: Desktop, history: List[Step]) -> str:
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


def image_to_b64(img: Image.Image, image_format: str = "PNG") -> str:
    """Converts a PIL Image to a base64-encoded string with MIME type included.

    Args:
        img (Image.Image): The PIL Image object to convert.
        image_format (str): The format to use when saving the image (e.g., 'PNG', 'JPEG').

    Returns:
        str: A base64-encoded string of the image with MIME type.
    """
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    image_data = buffer.getvalue()
    buffer.close()

    mime_type = f"image/{image_format.lower()}"
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def select_action(actions: List[ReasonedAction]) -> ReasonedAction:
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
