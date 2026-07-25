import queue
import threading
from typing import Any, Iterator


class PrefetchBuffer:
    """Asynchronously prefetches training batches in a background thread to prevent GPU starvation."""

    def __init__(self, dataloader_iterator: Iterator[Any], buffer_size: int = 4) -> None:
        self.dataloader_iterator = dataloader_iterator
        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def _prefetch(self) -> None:
        try:
            for batch in self.dataloader_iterator:
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
            self.queue.put(None)
        except Exception:
            self.queue.put(None)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        batch = self.queue.get()
        if batch is None:
            self.stop_event.set()
            raise StopIteration
        return batch

    def shutdown(self) -> None:
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=1.0)
