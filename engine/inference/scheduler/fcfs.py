from typing import Any, List

from engine.inference.scheduler.base import BaseScheduler


class FCFSScheduler(BaseScheduler):
    """First-Come-First-Serve request scheduler."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        return requests
