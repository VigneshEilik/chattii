from typing import Iterator

from engine.dataset.base.dataset import TextSample


class BaseLoader:
    """Abstract interface for streaming dataset readers."""

    def stream(self) -> Iterator[TextSample]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[TextSample]:
        return self.stream()
