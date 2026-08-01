import pytest

from engine.inference.streaming.streamer import (
    AsyncStreamer,
    CallbackStreamer,
    ConsoleStreamer,
)


def test_console_streamer():
    streamer = ConsoleStreamer()
    streamer.put("a")
    streamer.put("b")
    streamer.end()
    assert streamer.tokens == ["a", "b"]


def test_callback_streamer():
    received = []
    streamer = CallbackStreamer(callback=received.append)
    streamer.put("token1")
    streamer.put("token2")
    assert received == ["token1", "token2"]


@pytest.mark.anyio
async def test_async_streamer():
    streamer = AsyncStreamer()
    streamer.put("hello")
    streamer.put("world")
    streamer.end()

    collected = []
    async for tok in streamer:
        collected.append(tok)
    assert collected == ["hello", "world"]
