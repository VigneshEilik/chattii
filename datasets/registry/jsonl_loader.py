import hashlib
import json
import os
from typing import Any, Dict, Generator

from datasets.registry.base_loader import BaseTextLoader


class JsonlLoader(BaseTextLoader):
    """
    Ingests JSON Lines (.jsonl) files, streaming text segments from a targeted key.
    """

    def __init__(self, text_key: str = "text") -> None:
        self.text_key = text_key

    def load_lines(self, file_path: str) -> Generator[str, None, None]:
        """
        Reads a JSONL file and yields string fields line-by-line.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON at line {idx + 1}: {e}") from e

                if isinstance(data, dict) and self.text_key in data:
                    yield str(data[self.text_key])

    def get_provenance_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Calculates SHA-256 hash, byte size, and document count of a JSONL file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        sha256_hash = hashlib.sha256()
        line_count = 0

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    line_count += 1
                    sha256_hash.update(line.encode("utf-8"))

        return {
            "name": os.path.basename(file_path),
            "hash": sha256_hash.hexdigest(),
            "size_bytes": file_size,
            "encoding": "utf-8",
            "lines_count": line_count,
            "format": "jsonl",
        }
