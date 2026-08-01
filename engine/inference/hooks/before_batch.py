from typing import Any, List


class BeforeBatchHook:
    """Invoked prior to step request queue batch executions."""

    def __call__(self, batch: List[Any]) -> None:
        # Pre-execution hooks (e.g. logging)
        pass
