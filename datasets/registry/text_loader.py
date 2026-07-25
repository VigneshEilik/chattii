import hashlib
import os
from typing import Any, Dict, Generator

from datasets.registry.base_loader import BaseTextLoader


class TextLoader(BaseTextLoader):
    """
    Ingests plain-text files (.txt) line-by-line using generators.
    """

    def load_lines(self, file_path: str) -> Generator[str, None, None]:
        """
        Reads a raw text file and yields lines sequentially.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

    def get_provenance_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Calculates SHA-256 hash, byte size, and lines count of a text file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        sha256_hash = hashlib.sha256()
        line_count = 0
        file_size = os.path.getsize(file_path)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_count += 1
                sha256_hash.update(line.encode("utf-8"))

        return {
            "name": os.path.basename(file_path),
            "hash": sha256_hash.hexdigest(),
            "size_bytes": file_size,
            "encoding": "utf-8",
            "lines_count": line_count,
            "format": "txt",
        }
