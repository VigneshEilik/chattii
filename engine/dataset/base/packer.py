from typing import Iterator

from engine.dataset.base.dataset import TokenizedSample


class BasePacker:
    """Abstract interface for packing token sequences into fixed context windows."""

    def pack(self, samples: Iterator[TokenizedSample]) -> Iterator[TokenizedSample]:
        raise NotImplementedError
