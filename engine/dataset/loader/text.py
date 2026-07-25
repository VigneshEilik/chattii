import os
from typing import Iterator

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class TextLoader(BaseLoader):
    """Streams a plain text file line by line."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def stream(self) -> Iterator[TextSample]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Text source file not found: {self.path}")
        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            for idx, line in enumerate(f):
                cleaned = line.strip()
                if cleaned:
                    yield TextSample(
                        text=cleaned,
                        sample_id=f"txt_{idx}",
                        metadata={"line_number": idx, "source": self.path}
                    )
