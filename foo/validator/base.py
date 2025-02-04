from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from devicebay import Device
from rich.console import Console
from skillpacks import V1Action
from taskara import Task

from foo.actor.base import Step

console = Console(force_terminal=True)


T = TypeVar("T", bound=Device)


class ActionValidator(ABC, Generic[T]):
    """A validator that can validate a task action"""

    @abstractmethod
    def validate(
        self, action: V1Action, task: Task, device: T, history: List[Step]
    ) -> Step:
        pass
