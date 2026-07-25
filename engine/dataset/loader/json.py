# ruff: noqa: C901
import json
import os
from typing import Iterator

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class JsonLoader(BaseLoader):
    """Streams elements from a JSON array file."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def stream(self) -> Iterator[TextSample]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"JSON source file not found: {self.path}")

        # Fallback to standard json loading for compatibility
        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON file {self.path}: {e}")

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    text = ""
                    meta = {"index": idx, "source": self.path}
                    if isinstance(item, str):
                        text = item
                    elif isinstance(item, dict):
                        text = item.get("text", item.get("content", str(item)))
                        meta.update({k: v for k, v in item.items() if k not in ("text", "content")})
                    else:
                        text = str(item)

                    if text:
                        yield TextSample(
                            text=text,
                            sample_id=f"json_{idx}",
                            metadata=meta
                        )
            elif isinstance(data, dict):
                # If it's a dict, check for list values (e.g. {"data": [...]})
                for k, v in data.items():
                    if isinstance(v, list):
                        for idx, item in enumerate(v):
                            text = ""
                            meta = {"key": k, "index": idx, "source": self.path}
                            if isinstance(item, str):
                                text = item
                            elif isinstance(item, dict):
                                text = item.get("text", item.get("content", str(item)))
                                meta.update({kk: vv for kk, vv in item.items() if kk not in ("text", "content")})
                            else:
                                text = str(item)

                            if text:
                                yield TextSample(
                                    text=text,
                                    sample_id=f"json_{k}_{idx}",
                                    metadata=meta
                                )
                        break
            else:
                yield TextSample(
                    text=str(data),
                    sample_id="json_0",
                    metadata={"source": self.path}
                )
