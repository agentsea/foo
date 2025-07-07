import datetime
from dataclasses import dataclass
from typing import Any, List, Optional

from agentdesk import Desktop
from chatmux.openai import (
    ChatRequest,
    ChatResponse,
    ImageContentPart,
    ImageUrl,
    RequestMessage,
    SystemMessage,
    SystemMessageContent,
    SystemMessageContentPart,
    TextContentPart,
    UserMessage,
    UserMessageContent,
    UserMessageContentPart,
)
from pydantic import BaseModel
from skillpacks import EnvState, V1Action
from skillpacks.img import image_to_b64
from taskara import Task

from .action_parser import parse_action, translate_ad_action_to_qwen_action_dict


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
    end: bool = False
    result: Optional[str] = None
    raw_response: Optional[str] = None
    thought: Optional[str] = None
    note: Optional[str] = None
    description: Optional[str] = None
    in_tokens: int = 0
    out_tokens: int = 0


class ReasonedAction(BaseModel):
    reason: str
    note: str
    description: str
    action: V1Action


def parse_response(response: ChatResponse) -> List[ReasonedAction]:
    content = response.choices[0].message.content
    if content is None:
        return []
    parsed_action = parse_action(content)
    return [
        ReasonedAction(
            action=action,
            reason=parsed_action["thought"],
            description=parsed_action["description"],
            note=parsed_action["note"],
        )
        for action in parsed_action["actions"]
    ]

def build_scratchpad(descriptions: List[str], notes: list[str]) -> str:
    if len(descriptions) == 0:
        return "[Empty]"
    steps_str = "\n".join([f"* {description}" for description in descriptions])
    if len(notes) > 0:
        notes_str = "\n".join([f"* {note}" for note in notes if note != "None." or note != "None" or note != ""])
        return f"Steps I did so far:\n{steps_str}\nMy notes:\n{notes_str}\n"
    else:
        return f"Steps I did so far:\n{steps_str}\n"

def build_actor_messages_raw(
    task: Task, device: Desktop, history: List[Step]
) -> List[dict[str, Any | dict[str, Any]]]:
    messages: List[dict[str, Any | dict[str, Any]]] = []
    screenshots = device.take_screenshots(count=1)  # type: ignore
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt(task),
                },
                {
                    "type": "text",
                    "text": tools_list(),
                },
            ],
        }
    ]

    descriptions = [step.description for step in history if step.description is not None]
    notes = [step.note for step in history if step.note is not None]
    scratchpad = build_scratchpad(descriptions, notes)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Your scratchpad:\n\n{scratchpad}\n\nWhat is the next step?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_b64(screenshots[0])},  # type: ignore
                },
            ],
        }
    )
    return messages


def build_actor_messages_chatmux(
    task: Task, history: List[Step], gcs_url: str
) -> List[Any]:
    messages: List[Any] = []
    messages.append(
        SystemMessage(
            role="system",
            name=None,
            content=SystemMessageContent(
                root=[
                    SystemMessageContentPart(
                        root=TextContentPart(type="text", text=system_prompt(task))
                    ),
                    SystemMessageContentPart(
                        root=TextContentPart(type="text", text=tools_list())
                    ),
                ]
            ),
        )
    )

    descriptions = [step.description for step in history if step.description is not None]
    notes = [step.note for step in history if step.note is not None]
    scratchpad = build_scratchpad(descriptions, notes)
    messages.append(
        UserMessage(
            role="user",
            name=None,
            content=UserMessageContent(
                root=[
                    UserMessageContentPart(
                        root=TextContentPart(
                            type="text",
                            text=f"Your scratchpad:\n\n{scratchpad}\n\nWhat is the next step?",
                        )
                    ),
                    UserMessageContentPart(
                        root=ImageContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=gcs_url, detail="auto"),
                        )
                    ),
                ]
            ),
        )
    )
    return messages

#TODO: REBUILD THIS!!!
def create_actor_prompt_for_sft(
    task: Task,
    reason: str,
    description: str,
    note: str,
    action: V1Action,
    image_url: str,
    descriptions_history: List[str],
    notes_history: List[str],
) -> dict[str, Any]:
    scratchpad = build_scratchpad(descriptions_history, notes_history)

    response = f"""
{reason}
<note>
{note}
</note>
<next_action>
{description}
</next_action>
<tool_call>
{str(translate_ad_action_to_qwen_action_dict(action))}
</tool_call>
"""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt(task)},
                {"type": "text", "text": tools_list()},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Your scratchpad:\n\n{scratchpad}\n\nWhat is the next step?",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        },
    ]
    return {"messages": messages}


def system_prompt(task: Task) -> str:
    time = datetime.datetime.now().strftime("%A, %B %d, %Y")
    return f"""
<SYSTEM_CAPABILITY>
* You are a highly experienced Linux user, capable of using a mouse and keyboard to interact with a computer, and take screenshots.
* You are utilising an Linux virtual machine of screen size 1024x768 with internet access.
* To open Firefox, please just click on the web browser (globe) icon.
* The current date is {time}.
</SYSTEM_CAPABILITY>

<TASK>
{task.description}
</TASK>

<INSTRUCTIONS>
* You are given the task, your scratchpad (your notes and previously taken actions), and the screenshot of the current state. 
* You need to describe the current state, to consider the notes in your scratchpad, and to decide the next action. Also, describe what you expect to happen after the next action.
* Return your thoughts as plain text at the beginning of the response.
* Follow up with the note section: include in the note any information that is vital for your task and that you want to remember, for example, the information you were looking for and see on a screen now, contract information, etc. 
* Your note should not exceed one paragraph. 
* If there is nothing on the screen that you want to remember, just return 'None.' in the note section.
* Wrap the note in <note></note> XML tags.
* Follow up with a very brief description of the next action you will take, for example "Click on the 'Accept all' button", "Type 'New York' into the search bar", "Move the mouse to the 'Login' button", etc. 
* Wrap the description of the next action in <next_action></next_action> XML tags.
* Follow up with ONE corresponding function call. For a function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{{\"name\": \"<function-name>\", \"arguments\": \"<args-json-object>\"}}
</tool_call>

* You need ONLY ONE tool call for each action. DO NOT generate multiple tool calls. DO NOT repeat the same tool call in the same response.
* ALWAYS return the action in the format decribed above.
</INSTRUCTIONS>

<IMPORTANT>
* You are given the task and the action history with the current state screenshot. For each new screenshot, you need to describe the current state, to consider the previous actions, and to decide the next action. 
* When you open Google and see a cookie consent popup, click on the "Accept all" button (in any language). If the button is not visible, scroll down until you see it. Before scrolling, move the mouse closer to the header of the cookie consent popup.
* ALWAYS close the cookie consent popup on ANY website where you see it before continuing.
* You have a special action called `use_secret`. Use it to get the secret from the secret manager. You can ONLY use it when you are explicitly asked to do so. When you use it, you need to pass the name of the secret and the field of the secret. If you don't know the name of the field, you can use 'password' as the field.
</IMPORTANT>

<EXAMPLE>
I see that the population of Finland is 5.5 million. I need to close the browser now. I expect to see browser closed after the action.
<note>
The population of Finland is 5.5 million.
</note>
<next_action>
Close the browser.
</next_action>
<tool_call>
{{\"name\": \"computer_use\", \"arguments\": {{\"action\": \"left_click\", \"coordinate\": [100, 100]}}}}
</tool_call>
</EXAMPLE>
"""


def tools_list() -> str:
    return """

<TOOLS>
{"type": "function", "function": {
    "name_for_human": "computer_use",
    "name": "computer_use", 
    "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
                   "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
                   "You must click on desktop icons to start applications.\n"
                   "* Some applications may take time to start or process actions, so you may need to wait and take "
                   "successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window "
                   "doesn't open, try wait and taking another screenshot.\n"
                   "* The screen's resolution is 1024x768.\n"
                   "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
                   "screenshot to determine the coordinates of the element before moving the cursor.\n"
                   "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting "
                   "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
                   "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
                   "Don't click boxes on their edges unless asked.",
    "parameters": {
        "properties": {
            "action": {
                "description": "The action to perform. The available actions are:\n"
                             "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
                             "* `type`: Type a string of text on the keyboard.\n"
                             "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                             "* `left_click`: Click the left mouse button.\n"
                             "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                             "* `right_click`: Click the right mouse button.\n"
                             "* `middle_click`: Click the middle mouse button.\n"
                             "* `double_click`: Double-click the left mouse button.\n"
                             "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                             "* `use_secret`: Use a secret.\n"
                             "* `wait`: Wait specified seconds for the change to happen.\n"
                             "* `terminate`: Terminate the current task and report its completion status.",
                "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", 
                        "middle_click", "double_click", "scroll", "wait", "terminate"],
                "type": "string"
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array"
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string"
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. "
                             "Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array"
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. "
                             "Required only by `action=scroll`.",
                "type": "number"
            },
            "name": {
                "description": "The name of the secret to use. Required only by `action=use_secret`.",
                "type": "string"
            },
            "field": {
                "description": "The field of the secret to use. Required only by `action=use_secret`.",
                "type": "string"
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number"
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"]
            },
            "result": {
                "description": "The result of the task. Required only by `action=terminate`.",
                "type": "string"
            }
        },
        "required": ["action"],
        "type": "object"
    },
    "args_format": "Format the arguments as a JSON object."
}}
</TOOLS>
"""
