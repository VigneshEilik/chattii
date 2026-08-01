from typing import Any, List

from engine.inference.scheduler.base import BaseScheduler


class ContinuousBatchScheduler(BaseScheduler):
    """Continuous iteration/step scheduler stub."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        return requests
