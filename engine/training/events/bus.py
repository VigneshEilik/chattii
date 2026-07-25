# ruff: noqa: E501
from typing import Any, Callable, Dict, List


class EventBus:
    """A decoupled Event Bus supporting subscription and publishing of training states."""

    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable[..., None]]] = {}

    def subscribe(self, event_name: str, listener: Callable[..., None]) -> None:
        """Subscribes a listener function to a training event."""
        event_key = event_name.strip()
        if event_key not in self.subscribers:
            self.subscribers[event_key] = []
        self.subscribers[event_key].append(listener)

    def publish(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Publishes an event to all registered subscribers."""
        event_key = event_name.strip()
        if event_key in self.subscribers:
            for listener in self.subscribers[event_key]:
                listener(*args, **kwargs)
