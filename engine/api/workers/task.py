from typing import Any, Callable


class Task:
    """Encapsulates a unit of work to be scheduled in the background queue."""

    def __init__(
        self, task_id: str, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = "pending"

    def execute(self) -> Any:
        self.status = "running"
        try:
            result = self.func(*self.args, **self.kwargs)
            self.status = "completed"
            return result
        except Exception:
            self.status = "failed"
            raise
