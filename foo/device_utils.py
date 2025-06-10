from typing import Tuple

from toolfuse import Tool, action  # type: ignore


class DeviceUtils(Tool):
    """Common tool utilities for devices"""

    @action
    def end(self, result: str, comment: str) -> Tuple[str, str]:
        """End the task"""
        return result, comment
