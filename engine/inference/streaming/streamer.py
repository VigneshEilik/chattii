import asyncio
import sys
from typing import Any, Callable, List


class TokenStreamer:
    """Base streaming helper class returning generator iterators."""

    def __init__(self) -> None:
        self.tokens: List[str] = []

    def put(self, token: str) -> None:
        self.tokens.append(token)

    def end(self) -> None:
        pass


class ConsoleStreamer(TokenStreamer):
    """Streams generated tokens directly to stdout console logs."""

    def put(self, token: str) -> None:
        super().put(token)
        sys.stdout.write(token)
        sys.stdout.flush()

    def end(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


class CallbackStreamer(TokenStreamer):
    """Executes registered callback functions when new tokens are yielded."""

    def __init__(self, callback: Callable[[str], Any]) -> None:
        super().__init__()
        self.callback = callback

    def put(self, token: str) -> None:
        super().put(token)
        self.callback(token)


class AsyncStreamer(TokenStreamer):
    """Asynchronous queue iterator for Server-Sent Events (SSE) interfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: asyncio.Queue = asyncio.Queue()

    def put(self, token: str) -> None:
        super().put(token)
        self.queue.put_nowait(token)

    def end(self) -> None:
        self.queue.put_nowait(None)

    def __aiter__(self) -> "AsyncStreamer":
        return self

    async def __anext__(self) -> str:
        token = await self.queue.get()
        if token is None:
            raise StopAsyncIteration
        return token
