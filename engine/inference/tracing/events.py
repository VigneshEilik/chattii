import time
from typing import Any, Dict, List, Optional


class EventLogger:
    """Logs discrete checkpoint events with key-value payloads."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            {"name": name, "timestamp": time.perf_counter(), "payload": payload or {}}
        )
