import json
import os
from typing import Iterator

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class JsonlLoader(BaseLoader):
    """Streams JSON line-by-line objects (JSONL)."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def stream(self) -> Iterator[TextSample]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"JSONL source file not found: {self.path}")

        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    # Ignore invalid lines, print or log warning
                    continue

                text = ""
                meta = {"line": idx, "source": self.path}
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text", item.get("content", str(item)))
                    meta.update(
                        {k: v for k, v in item.items() if k not in ("text", "content")}
                    )
                else:
                    text = str(item)

                if text:
                    yield TextSample(text=text, sample_id=f"jsonl_{idx}", metadata=meta)
