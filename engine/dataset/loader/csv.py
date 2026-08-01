import csv
import os
from typing import Iterator

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader


class CsvLoader(BaseLoader):
    """Streams rows from a CSV file."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def stream(self) -> Iterator[TextSample]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV source file not found: {self.path}")

        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            reader = csv.reader(f)
            # Try to read header to inspect fields
            try:
                header = next(reader)
            except StopIteration:
                return

            text_col_idx = 0
            # If there's a column named "text" or "content", use that column index
            for idx, col in enumerate(header):
                if col.lower() in ("text", "content"):
                    text_col_idx = idx
                    break

            for idx, row in enumerate(reader):
                if not row:
                    continue
                # If row has fewer elements, fallback
                if text_col_idx < len(row):
                    text = row[text_col_idx]
                else:
                    text = row[0] if row else ""

                if text:
                    meta = {
                        "row_index": idx,
                        "source": self.path,
                    }
                    # Include key-value headers in metadata if row matches header len
                    if len(row) == len(header):
                        meta.update(
                            {
                                header[i]: row[i]
                                for i in range(len(header))
                                if i != text_col_idx
                            }
                        )

                    yield TextSample(text=text, sample_id=f"csv_{idx}", metadata=meta)
