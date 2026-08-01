import json
import os
from typing import Any, Dict


class DistributedCheckpointMetadata:
    """Manages distributed checkpoint metadata indexing files."""

    @staticmethod
    def write_metadata(
        path: str, world_size: int, strategy: str, files: Dict[int, str]
    ) -> None:
        metadata = {
            "world_size": world_size,
            "strategy": strategy,
            "shard_mapping": files,
        }
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def read_metadata(path: str) -> Dict[str, Any]:
        metadata_file = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_file):
            return {}
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
