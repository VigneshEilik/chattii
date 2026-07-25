import json
import os
from typing import Any, Dict


class PrepCheckpointManager:
    """Tracks state checkpoints during dataset tokenization and sharding to support resume features."""

    def __init__(self, filepath: str = ".cache/dataset/prep_checkpoint.json") -> None:
        self.filepath = filepath
        self.state: Dict[str, Any] = {
            "processed_files": [],
            "current_file": "",
            "file_offset": 0,
            "sample_index": 0,
            "token_count": 0,
            "shard_index": 0,
        }
        self.load()

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                pass
        else:
            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def update(
        self,
        current_file: str,
        file_offset: int,
        sample_index: int,
        token_count: int,
        shard_index: int,
    ) -> None:
        self.state["current_file"] = current_file
        self.state["file_offset"] = file_offset
        self.state["sample_index"] = sample_index
        self.state["token_count"] = token_count
        self.state["shard_index"] = shard_index

        if current_file and current_file not in self.state["processed_files"]:
            self.state["processed_files"].append(current_file)
        self.save()

    def clear(self) -> None:
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass
        self.state = {
            "processed_files": [],
            "current_file": "",
            "file_offset": 0,
            "sample_index": 0,
            "token_count": 0,
            "shard_index": 0,
        }
