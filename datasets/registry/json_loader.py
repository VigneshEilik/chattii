import hashlib
import json
import os
from typing import Any, Dict, Generator

from datasets.registry.base_loader import BaseTextLoader


class JsonLoader(BaseTextLoader):
    """
    Ingests JSON files containing arrays of string documents.
    """

    def load_lines(self, file_path: str) -> Generator[str, None, None]:
        """
        Loads the JSON array and yields elements sequentially.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file format: {e}") from e

            if not isinstance(data, list):
                raise ValueError("JSON dataset must contain a flat array of strings.")

            for document in data:
                yield str(document)

    def get_provenance_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Calculates SHA-256 hash, byte size, and document count of a JSON array file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        sha256_hash = hashlib.sha256()
        line_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            sha256_hash.update(content.encode("utf-8"))
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    line_count = len(data)
            except json.JSONDecodeError:
                pass

        return {
            "name": os.path.basename(file_path),
            "hash": sha256_hash.hexdigest(),
            "size_bytes": file_size,
            "encoding": "utf-8",
            "lines_count": line_count,
            "format": "json",
        }
