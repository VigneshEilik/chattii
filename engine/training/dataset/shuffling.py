# ruff: noqa: E501
import random
from typing import Iterator, List, TypeVar

T = TypeVar("T")


class BufferShuffler:
    """Performs stream-based pseudo-random shuffling using a memory buffer."""

    @staticmethod
    def shuffle(iterator: Iterator[T], buffer_size: int = 1000) -> Iterator[T]:
        """
        Yields items from iterator after buffer-based shuffling to limit memory footprint.
        """
        buffer: List[T] = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Empty the remaining items randomly
        random.shuffle(buffer)
        for item in buffer:
            yield item
