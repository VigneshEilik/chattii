import asyncio
import logging

from engine.api.workers.queue import TaskQueue

logger = logging.getLogger("chatii.workers")


class QueueWorker:
    """Thread execution loop pulling tasks and executing them in order."""

    def __init__(self, queue: TaskQueue) -> None:
        self.queue = queue
        self.running = False
        self._task = None

    def start(self) -> None:
        self.running = True
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while self.running:
            try:
                task = await self.queue.get()
                logger.info(f"Starting task execution: {task.task_id}")

                # Execute blocking code inside run_in_executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, task.execute)

                logger.info(f"Finished task execution: {task.task_id}")
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task failed: {e}")
                self.queue.task_done()
