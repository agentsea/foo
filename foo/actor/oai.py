from typing import List, Optional

from agentdesk import Desktop
from json_repair import repair_json
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from skillpacks.img import image_to_b64
from taskara import Task
from threadmem import RoleThread

from .base import Actor, Step

console = Console(force_terminal=True)


class OaiActor(Actor[Desktop]):
    """An actor that uses fine tuned openai models"""

    def __init__(self, model: Optional[str] = None):
        self.model = "ft:gpt-4o-2024-08-06:kentauros-ai:bookbnb2:Amc6oDGF"
        self.client = OpenAI()

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        thread = RoleThread()

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant operating a computer.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_b64(screenshots[0]),
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": f"You are given a task to complete: '{task.description}'\n"
                + 'You have the following tools at your disposal: [{\'name\': \'click\', \'parameters\': {\'type\': \'object\', \'properties\': {\'button\': {\'type\': \'string\', \'description\': \'Button to click. Defaults to "left".\'}, \'x\': {\'anyOf\': [{\'type\': \'integer\'}, {\'type\': \'null\'}], \'description\': \'X coordinate to move to, if not provided it will click on current location. Defaults to None.\'}, \'y\': {\'anyOf\': [{\'type\': \'integer\'}, {\'type\': \'null\'}], \'description\': \'Y coordinate to move to, if not provided it will click on current location. Defaults to None.\'}}, \'required\': []}, \'description\': \'Click mouse button\'}, {\'name\': \'double_click\', \'parameters\': {\'type\': \'object\', \'properties\': {}, \'required\': []}, \'description\': \'Double click the mouse\'}, {\'name\': \'drag_mouse\', \'parameters\': {\'type\': \'object\', \'properties\': {\'x\': {\'type\': \'integer\', \'description\': \'x coordinate\'}, \'y\': {\'type\': \'integer\', \'description\': \'y coordinate\'}}, \'required\': [\'x\', \'y\']}, \'description\': \'Drag the mouse\'}, {\'name\': \'exec\', \'parameters\': {\'type\': \'object\', \'properties\': {\'cmd\': {\'type\': \'string\', \'description\': \'Command to execute\'}}, \'required\': [\'cmd\']}, \'description\': \'Execute a command\'}, {\'name\': \'hot_key\', \'parameters\': {\'type\': \'object\', \'properties\': {\'keys\': {\'type\': \'array\', \'items\': {\'type\': \'string\'}, \'description\': \'Which keys to press. Options are:\\n[ "\\\\t", "\\\\n", "\\\\r", " ", "!", \\\'\\\\"\\\', "\\\\#", "\\\\$", "\\\\%", "\\\\&", "\\\\\\\'",\\n"\\\\(", "\\\\)", "\\\\*", "\\\\+", ",", "-", "\\\\.", "/", "0", "1", "2", "3",\\n"4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "\\\\?", "@",\\n"\\\\[", "\\\\\\\\", "\\\\]", "\\\\^", "\\\\_", "\\\\`", "a", "b", "c", "d", "e",\\n"f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",\\n"t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "accept", "add",\\n"alt", "altleft", "altright", "apps", "backspace", "browserback",\\n"browserfavorites", "browserforward", "browserhome", "browserrefresh",\\n"browsersearch", "browserstop", "capslock", "clear", "convert", "ctrl",\\n"ctrlleft", "ctrlright", "decimal", "del", "delete", "divide", "down",\\n"end", "enter", "esc", "escape", "execute", "f1", "f10", "f11", "f12",\\n"f13", "f14", "f15", "f16", "f17", "f18", "f19", "f2", "f20", "f21",\\n"f22", "f23", "f24", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "final",\\n"fn", "help", "home", "insert", "left", "numlock", "pagedown", "pageup", "pause",\\n"pgdn", "pgup", "playpause", "prevtrack", "print", "printscreen",\\n"prntscrn", "prtsc", "prtscr", "return", "right", "scrolllock",\\n"select", "separator", "shift", "shiftleft", "shiftright", "sleep",\\n"space", "stop", "subtract", "tab", "up", "volumedown", "volumemute",\\n"volumeup", "win", "winleft", "winright", "yen", "command", "option",\\n"optionleft", "optionright" ]\'}}, \'required\': [\'keys\']}, \'description\': \'Press a hot key. For example ctrl+c\'}, {\'name\': \'move_mouse\', \'parameters\': {\'type\': \'object\', \'properties\': {\'x\': {\'type\': \'integer\', \'description\': \'x coordinate\'}, \'y\': {\'type\': \'integer\', \'description\': \'y coordinate\'}}, \'required\': [\'x\', \'y\']}, \'description\': \'Move mouse to a position\'}, {\'name\': \'open_url\', \'parameters\': {\'type\': \'object\', \'properties\': {\'url\': {\'type\': \'string\', \'description\': \'URL to open\'}}, \'required\': [\'url\']}, \'description\': \'Open a URL in chromium\'}, {\'name\': \'press_key\', \'parameters\': {\'type\': \'object\', \'properties\': {\'key\': {\'type\': \'string\', \'description\': \'Which key to press. Options are:\\n[ "\\\\t", "\\\\n", "\\\\r", " ", "!", \\\'\\\\"\\\', "\\\\#", "\\\\$", "\\\\%", "\\\\&", "\\\\\\\'",\\n"\\\\(", "\\\\)", "\\\\*", "\\\\+", ",", "-", "\\\\.", "/", "0", "1", "2", "3",\\n"4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "\\\\?", "@",\\n"\\\\[", "\\\\\\\\", "\\\\]", "\\\\^", "\\\\_", "\\\\`", "a", "b", "c", "d", "e",\\n"f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",\\n"t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "accept", "add",\\n"alt", "altleft", "altright", "apps", "backspace", "browserback",\\n"browserfavorites", "browserforward", "browserhome", "browserrefresh",\\n"browsersearch", "browserstop", "capslock", "clear", "convert", "ctrl",\\n"ctrlleft", "ctrlright", "decimal", "del", "delete", "divide", "down",\\n"end", "enter", "esc", "escape", "execute", "f1", "f10", "f11", "f12",\\n"f13", "f14", "f15", "f16", "f17", "f18", "f19", "f2", "f20", "f21",\\n"f22", "f23", "f24", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "final",\\n"fn", "help", "home", "insert", "left", "numlock", "pagedown", "pageup", "pause",\\n"pgdn", "pgup", "playpause", "prevtrack", "print", "printscreen",\\n"prntscrn", "prtsc", "prtscr", "return", "right", "scrolllock",\\n"select", "separator", "shift", "shiftleft", "shiftright", "sleep",\\n"space", "stop", "subtract", "tab", "up", "volumedown", "volumemute",\\n"volumeup", "win", "winleft", "winright", "yen", "command", "option",\\n"optionleft", "optionright" ]\'}}, \'required\': [\'key\']}, \'description\': \'Press a key\'}, {\'name\': \'scroll\', \'parameters\': {\'type\': \'object\', \'properties\': {\'clicks\': {\'type\': \'integer\', \'description\': \'Number of clicks, negative scrolls down, positive scrolls up. Defaults to -3.\'}}, \'required\': []}, \'description\': \'Scroll the screen\'}, {\'name\': \'type_text\', \'parameters\': {\'type\': \'object\', \'properties\': {\'text\': {\'type\': \'string\', \'description\': \'Text to type\'}}, \'required\': [\'text\']}, \'description\': \'Type text\'}, {\'name\': \'mouse_coordinates\', \'parameters\': {\'type\': \'object\', \'properties\': {}, \'required\': []}, \'description\': \'Get the current mouse coordinates\'}, {\'name\': \'take_screenshots\', \'parameters\': {\'type\': \'object\', \'properties\': {\'count\': {\'type\': \'integer\'}, \'delay\': {\'type\': \'number\'}}, \'required\': []}, \'description\': \'Take screenshots\'}]\n\nI\'ve provided you with the latest screenshot of the computer.\n\nPlease return the action you want to take as a raw JSON object.',
            },
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
        )

        try:
            actions = self._parse_response(completion)
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

        step = Step(
            state=EnvState(images=screenshots),
            action=selection,
            task=task,
            thread=thread,
            model_id=self.model,
        )

        return step

    def _parse_response(self, response: ChatCompletion) -> List[V1Action]:
        output = []
        for choice in response.choices:
            try:
                obj = repair_json(choice.message.content, return_objects=True)  # type: ignore
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
