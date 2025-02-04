from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from devicebay import Device
from orign import V1ChatEvent
from orign.models import MessageItem
from rich.console import Console
from skillpacks import EnvState, V1Action
from taskara import Task

console = Console(force_terminal=True)


@dataclass
class Step:
    """A step in an episode"""

    state: EnvState
    action: V1Action
    action_opts: Optional[List[V1Action]] = None
    thread: Optional[List[MessageItem]] = None
    task: Optional[Task] = None
    model_id: Optional[str] = None
    prompt: Optional[V1ChatEvent] = None


T = TypeVar("T", bound=Device)


class Actor(ABC, Generic[T]):
    """An actor that can act on a task"""

    @abstractmethod
    def act(self, task: Task, device: T, history: List[Step]) -> Step:
        pass

    @abstractmethod
    def get_ctx(self, task: Task, device: T, history: List[Step]) -> str:
        pass
