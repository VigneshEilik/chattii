from typing import Any, List

from engine.inference.scheduler.base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """Round-robin request scheduler stub."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        return requests
