from typing import Any, Optional


class ApplicationState:
    """Stateful context object holding service registries, managers, and performance configurations."""

    def __init__(self) -> None:
        self.model_manager: Optional[Any] = None
        self.session_manager: Optional[Any] = None
        self.metrics_tracker: Optional[Any] = None
        self.config: Optional[Any] = None
        self.task_dispatcher: Optional[Any] = None
        self.limiter: Optional[Any] = None
        self.tracer: Optional[Any] = None
