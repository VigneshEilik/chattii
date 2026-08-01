from typing import Any, List

from engine.inference.scheduler.base import BaseScheduler


class PrefillDecodeScheduler(BaseScheduler):
    """Prefill vs. Decode iteration scheduler stub."""

    def schedule(self, requests: List[Any]) -> List[Any]:
        # Divide requests into prefill phase and decode phase (stub returns all)
        return requests
