from typing import Any, List


class BaseScheduler:
    """Base interface for coordinating generation batch requests."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        raise NotImplementedError
