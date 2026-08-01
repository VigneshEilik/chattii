from typing import Any, List

from engine.inference.scheduler.base import BaseScheduler


class PriorityScheduler(BaseScheduler):
    """Schedules requests based on user priority weights."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        # Sort requests based on request priority parameter (higher value first)
        return sorted(requests, key=lambda r: getattr(r, "priority", 0), reverse=True)
