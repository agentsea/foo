# type: ignore

import re
from typing import Any

import json_repair
from rich.console import Console
from skillpacks.server.models import V1Action

console = Console()


def parse_action(content: str) -> dict[str, Any]:
    """
    "action":
    * `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
    * `type`: Type a string of text on the keyboard.
    * `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
    * `left_click`: Click the left mouse button.
    * `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
    * `right_click`: Click the right mouse button.
    * `middle_click`: Click the middle mouse button.
    * `double_click`: Double-click the left mouse button.
    * `scroll`: Performs a scroll of the mouse scroll wheel.
    * `wait`: Wait specified seconds for the change to happen.
    * `terminate`: Terminate the current task and report its completion status.

    "parameters":
        "keys": {
            "description": "Required only by `action=key`.",
            "type": "array",
        },
        "text": {
            "description": "Required only by `action=type`.",
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
            "type": "array",
        },
        "pixels": {
            "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
            "type": "number",
        },
        "time": {
            "description": "The seconds to wait. Required only by `action=wait`.",
            "type": "number",
        },
        "status": {
            "description": "The status of the task. Required only by `action=terminate`.",
            "type": "string",
            "enum": ["success", "failure"],
        },
        "result": {
            "description": "The result of the task. Required only by `action=terminate`.",
            "type": "string",
        },
    }

        Example:
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1240, 783]}}
        </tool_call>

    Extra actions:
    * `use_secret`: Use a secret. Parameters: `name`, `field`.
    """

    output = []
    console.print(f"Raw content: {content}")

    # Extract any text before the scratchpad as thought
    pre_scratchpad_pattern = r"^(.*?)(?=<scratchpad>)"
    pre_scratchpad_match = re.search(pre_scratchpad_pattern, content, re.DOTALL)
    if pre_scratchpad_match:
        thought = pre_scratchpad_match.group(1).strip()
    else:
        thought = ""

    # Extract scratchpad between <scratchpad> and </scratchpad> tags
    note_pattern = r"<note>\n(.*?)\n(?:</note>)"
    note_match = re.search(note_pattern, content, re.DOTALL)
    if note_match:
        note = note_match.group(1).strip()
    else:
        note = ""

    # Extract next action between <next_action> and </next_action> tags
    next_action_pattern = r"<next_action>\n(.*?)\n(?:</next_action>)"
    next_action_match = re.search(next_action_pattern, content, re.DOTALL)
    if next_action_match:
        next_action = next_action_match.group(1).strip()
    else:
        next_action = ""

    # Extract tool calls between <tool_call> and </tool_call> tags
    tool_call_pattern = r"<tool_call>\n(.*?)\n(?:</tool_call>|📐|⚗)"
    tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)
    tools_used = []
    if tool_call_matches:
        for match in tool_call_matches:
            tools_used.append(match.strip())

    for tool_used in tools_used:
        tool_used_json = json_repair.loads(tool_used)
        console.print(f"Found tool usage: {tool_used_json}", style="green")
        action_name = tool_used_json["arguments"]["action"]
        parameters = {}

        if action_name == "key":
            action_name = "hot_key"
            parameters["keys"] = tool_used_json["arguments"]["keys"]
        elif action_name == "type":
            action_name = "type_text"
            parameters["text"] = tool_used_json["arguments"]["text"]
        elif action_name == "mouse_move":
            action_name = "move_mouse"
            parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
            parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "left_click":
            action_name = "click"
            parameters["button"] = "left"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "left_click_drag":
            action_name = "drag_mouse"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "right_click":
            action_name = "click"
            parameters["button"] = "right"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "middle_click":
            action_name = "click"
            parameters["button"] = "middle"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "double_click":
            action_name = "double_click"
            parameters["button"] = "left"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "scroll":
            action_name = "scroll"
            parameters["clicks"] = tool_used_json["arguments"]["pixels"] // 10
        elif action_name == "use_secret":
            action_name = "use_secret"
            parameters["name"] = tool_used_json["arguments"]["name"]
            parameters["field"] = tool_used_json["arguments"]["field"]
        elif action_name == "wait":
            action_name = "wait"
            parameters["seconds"] = tool_used_json["arguments"]["time"]
        elif action_name == "terminate":
            action_name = "end"
            if "result" not in tool_used_json["arguments"]:
                tool_used_json["arguments"]["result"] = "I'm done!"
            parameters["result"] = tool_used_json["arguments"]["result"]
            parameters["comment"] = thought

        console.print(f"Parsed Action: {action_name}", style="yellow")
        console.print(f"Parsed Params: {parameters}", style="blue")

        # Create the V1Action
        action = V1Action(name=action_name, parameters=parameters)
        output.append(action)

    return {
        "thought": thought,
        "note": note,
        "description": next_action,
        "actions": output,
    }


def translate_ad_action_to_qwen_action_dict(action: V1Action) -> dict[str, Any]:
    """
    Translate an AD action to a Qwen action dict.
    E.g. to {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1240, 783]}}
    """
    name = action.name
    parameters = action.parameters
    if name == "hot_key":
        return {
            "name": "computer_use",
            "arguments": {"action": "key", "keys": parameters["keys"]},
        }
    elif name == "press_key":
        return {
            "name": "computer_use",
            "arguments": {"action": "key", "keys": [parameters["key"]]},
        }
    elif name == "type_text":
        return {
            "name": "computer_use",
            "arguments": {"action": "type", "text": parameters["text"]},
        }
    elif name == "move_mouse":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "mouse_move",
                "coordinate": [parameters["x"], parameters["y"]],
            },
        }
    elif name == "click":
        if parameters["button"] == "left":
            return {
                "name": "computer_use",
                "arguments": {
                    "action": "left_click",
                    "coordinate": [parameters["x"], parameters["y"]],
                },
            }
        elif parameters["button"] == "right":
            return {
                "name": "computer_use",
                "arguments": {
                    "action": "right_click",
                    "coordinate": [parameters["x"], parameters["y"]],
                },
            }
        elif parameters["button"] == "middle":
            return {
                "name": "computer_use",
                "arguments": {
                    "action": "middle_click",
                    "coordinate": [parameters["x"], parameters["y"]],
                },
            }
    elif name == "drag_mouse":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "left_click_drag",
                "coordinate": [parameters["x"], parameters["y"]],
            },
        }
    elif name == "double_click":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "double_click",
                "button": "left",
                "coordinate": [parameters["x"], parameters["y"]],
            },
        }
    elif name == "scroll":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "scroll",
                "pixels": parameters["clicks"] * 10,
            },
        }
    elif name == "use_secret":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "use_secret",
                "name": parameters["name"],
                "field": parameters["field"],
            },
        }
    elif name == "wait":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "wait",
                "time": parameters["seconds"] if "seconds" in parameters else 3,
            },
        }
    elif name == "result":
        # TODO: remove this when we don't have old skills trained with `result` any longer
        # (the action space change is introduced in 2025-06-09)
        return {
            "name": "computer_use",
            "arguments": {
                "action": "terminate",
                "status": "success",
                "result": parameters["value"],
            },
        }
    elif name == "end":
        return {
            "name": "computer_use",
            "arguments": {
                "action": "terminate",
                "status": "success",
                "result": parameters["result"]
                if "result" in parameters
                else "I'm done!",
            },
        }
    else:
        console.print(f"Unknown action: {name}; falling back to wait.", style="red")
        return {
            "name": "computer_use",
            "arguments": {
                "action": "wait",
                "time": 1,
            },
        }


if __name__ == "__main__":
    # Example 1: Key Action
    print("=== Example 1: key action ===")
    content_key = (
        "Thought: Let's press some keys\n"
        "<scratchpad>\n"
        "I've pressed ctrl+v\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "Pressed ctrl+c\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "c"]}}\n'
        "</tool_call>"
    )
    actions_key = parse_action(content_key)
    print("Actions (key):", actions_key)
    print("")

    # Example 2: Type Action
    print("=== Example 2: type action ===")
    content_type = (
        "Thought: I'll type some text\n"
        "<scratchpad>\n"
        "I'll type Hello World\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've typed Hello World\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "type", "text": "Hello World"}}\n'
        "</tool_call>"
    )
    actions_type = parse_action(content_type)
    print("Actions (type):", actions_type)
    print("")

    # Example 3: Mouse Move Action
    print("=== Example 3: mouse move action ===")
    content_move = (
        "Thought: Moving the mouse\n"
        "<scratchpad>\n"
        "I'll move the mouse to (100, 200)\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've moved the mouse to (100, 200)\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": [100, 200]}}\n'
        "</tool_call>"
    )
    actions_move = parse_action(content_move)
    print("Actions (move):", actions_move)
    print("")

    # Example 4: Left Click Action
    print("=== Example 4: left click action ===")
    content_click = (
        "Thought: Let's click something\n"
        "<scratchpad>\n"
        "I'll click the left mouse button\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've clicked the left mouse button\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click"}}\n'
        "</tool_call>"
    )
    actions_click = parse_action(content_click)
    print("Actions (click):", actions_click)
    print("")

    # Example 5: Drag Action
    print("=== Example 5: drag action ===")
    content_drag = (
        "Thought: I'm dragging something\n"
        "<scratchpad>\n"
        "I'll drag the mouse to (300, 400)\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've dragged the mouse to (300, 400)\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click_drag", "coordinate": [300, 400]}}\n'
        "</tool_call>"
    )
    actions_drag = parse_action(content_drag)
    print("Actions (drag):", actions_drag)
    print("")

    # Example 6: Right Click Action
    print("=== Example 6: right click action ===")
    content_right = (
        "Thought: Right clicking\n"
        "<scratchpad>\n"
        "I'll click the right mouse button\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've clicked the right mouse button\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "right_click"}}\n'
        "</tool_call>"
    )
    actions_right = parse_action(content_right)
    print("Actions (right):", actions_right)
    print("")

    # Example 7: Middle Click Action
    print("=== Example 7: middle click action ===")
    content_middle = (
        "Thought: Middle clicking\n"
        "<scratchpad>\n"
        "I'll click the middle mouse button\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've clicked the middle mouse button\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "middle_click"}}\n'
        "</tool_call>"
    )
    actions_middle = parse_action(content_middle)
    print("Actions (middle):", actions_middle)
    print("")

    # Example 8: Double Click Action
    print("=== Example 8: double click action ===")
    content_double = (
        "Thought: Double clicking\n"
        "<scratchpad>\n"
        "I'll double click the left mouse button\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've double clicked the left mouse button\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "double_click"}}\n'
        "</tool_call>"
    )
    actions_double = parse_action(content_double)
    print("Actions (double):", actions_double)
    print("")

    # Example 9: Scroll Action
    print("=== Example 9: scroll action ===")
    content_scroll = (
        "Thought: Scrolling down\n"
        "<scratchpad>\n"
        "I'll scroll down\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've scrolled down\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "scroll", "pixels": -30}}\n'
        "</tool_call>"
    )
    actions_scroll = parse_action(content_scroll)
    print("Actions (scroll):", actions_scroll)
    print("")

    # Example 10: Wait Action
    print("=== Example 10: wait action ===")
    content_wait = (
        "Thought: Waiting for a bit\n"
        "<scratchpad>\n"
        "I'll wait for 5 seconds\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I've waited for 5 seconds\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "wait", "time": 5}}\n'
        "</tool_call>"
    )
    actions_wait = parse_action(content_wait)
    print("Actions (wait):", actions_wait)
    print("")

    # Example 11: Terminate Action
    print("=== Example 11: terminate action ===")
    content_terminate = (
        "Thought: Task completed successfully\n"
        "<scratchpad>\n"
        "I'm done!\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I'm done!\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success", "result": "I\'m done!"}}\n'
        "</tool_call>"
    )
    actions_terminate = parse_action(content_terminate)
    print("Actions (terminate):", actions_terminate)
    print("")

    # Example 12: Incomplete Terminate Action
    print("=== Example 12: incomplete terminate action ===")
    content_terminate = (
        "Thought: Task completed successfully\n"
        "<scratchpad>\n"
        "I'm done!\n"
        "</scratchpad>\n"
        "<next_action>\n"
        "I'm done!\n"
        "</next_action>\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
        "</tool_call>"
    )
    actions_terminate = parse_action(content_terminate)
    print("Actions (terminate):", actions_terminate)
    print("")

    # Example 13: Translate AD actions back to Qwen actions
    print("=== Set of examples: AD to Qwen action translation ===")

    # Hot key action
    ad_hot_key = V1Action(name="hot_key", parameters={"keys": ["ctrl", "c"]})
    qwen_hot_key = translate_ad_action_to_qwen_action_dict(ad_hot_key)
    print("Hot key translation:", qwen_hot_key)
    print("")

    # Type text action
    ad_type = V1Action(name="type_text", parameters={"text": "Hello world"})
    qwen_type = translate_ad_action_to_qwen_action_dict(ad_type)
    print("Type text translation:", qwen_type)
    print("")

    # Move mouse action
    ad_move = V1Action(name="move_mouse", parameters={"x": 100, "y": 200})
    qwen_move = translate_ad_action_to_qwen_action_dict(ad_move)
    print("Move mouse translation:", qwen_move)
    print("")

    # Click actions
    ad_left_click = V1Action(
        name="click", parameters={"button": "left", "x": 100, "y": 200}
    )
    qwen_left_click = translate_ad_action_to_qwen_action_dict(ad_left_click)
    print("Left click translation:", qwen_left_click)

    ad_right_click = V1Action(
        name="click", parameters={"button": "right", "x": 100, "y": 200}
    )
    qwen_right_click = translate_ad_action_to_qwen_action_dict(ad_right_click)
    print("Right click translation:", qwen_right_click)

    ad_middle_click = V1Action(
        name="click", parameters={"button": "middle", "x": 100, "y": 200}
    )
    qwen_middle_click = translate_ad_action_to_qwen_action_dict(ad_middle_click)
    print("Middle click translation:", qwen_middle_click)
    print("")

    # Drag mouse action
    ad_drag = V1Action(name="drag_mouse", parameters={"x": 100, "y": 200})
    qwen_drag = translate_ad_action_to_qwen_action_dict(ad_drag)
    print("Drag mouse translation:", qwen_drag)
    print("")

    # Double click action
    ad_double = V1Action(name="double_click", parameters={"x": 100, "y": 200})
    qwen_double = translate_ad_action_to_qwen_action_dict(ad_double)
    print("Double click translation:", qwen_double)
    print("")

    # Scroll action
    ad_scroll = V1Action(name="scroll", parameters={"clicks": 3})
    qwen_scroll = translate_ad_action_to_qwen_action_dict(ad_scroll)
    print("Scroll translation:", qwen_scroll)
    print("")

    # Wait action
    ad_wait = V1Action(name="wait", parameters={"seconds": 5})
    qwen_wait = translate_ad_action_to_qwen_action_dict(ad_wait)
    print("Wait translation:", qwen_wait)
    print("")

    # Result action
    ad_result = V1Action(name="result", parameters={"value": "Task completed!"})
    qwen_result = translate_ad_action_to_qwen_action_dict(ad_result)
    print("Result translation:", qwen_result)
    print("")
