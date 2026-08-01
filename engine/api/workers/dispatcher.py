from typing import List

from engine.api.workers.queue import TaskQueue
from engine.api.workers.task import Task
from engine.api.workers.worker import QueueWorker


class TaskDispatcher:
    """Manages pool of background QueueWorkers and delegates incoming Tasks."""

    def __init__(self, pool_size: int = 2) -> None:
        self.queue = TaskQueue()
        self.workers: List[QueueWorker] = [
            QueueWorker(self.queue) for _ in range(pool_size)
        ]

    def start(self) -> None:
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        for worker in self.workers:
            worker.stop()

    def submit_task(self, task: Task) -> None:
        self.queue.put(task)
