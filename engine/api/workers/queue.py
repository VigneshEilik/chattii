import asyncio

from engine.api.workers.task import Task


class TaskQueue:
    """Thread-safe queue wrapper managing asynchronous task pipelines."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    def put(self, task: Task) -> None:
        self.queue.put_nowait(task)

    async def get(self) -> Task:
        return await self.queue.get()

    def task_done(self) -> None:
        self.queue.task_done()

    def size(self) -> int:
        return self.queue.qsize()
