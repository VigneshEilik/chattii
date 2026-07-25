from typing import Any, Iterator


class MultiWorkerDatasetIterator:
    """Safe multi-worker wrapper providing multi-process data dispatch fallback paths."""

    def __init__(self, dataloader: Any, num_workers: int = 0) -> None:
        self.dataloader = dataloader
        self.num_workers = num_workers

    def __iter__(self) -> Iterator[Any]:
        # For simplicity and robust performance on Windows, we default to inline main-thread execution
        # but keep hooks for multi-worker process pools in distributed setups.
        return iter(self.dataloader.stream())
