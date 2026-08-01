import time
from typing import Any, Dict, List


class DistributedTimelineProfiler:
    """Collects timestamps of step events for profiling timelines."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record_event(self, name: str) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
            }
        )

    def get_timeline(self) -> List[Dict[str, Any]]:
        return self.events
