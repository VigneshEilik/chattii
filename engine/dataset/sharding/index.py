import hashlib
import json
import os
from typing import Any, Dict, List


class ShardIndexGenerator:
    """Generates index.json mapping offsets, lengths, token counts, and checksums of binary shards."""

    @staticmethod
    def generate_index(shard_paths: List[str], dtype_size: int = 2) -> Dict[str, Any]:
        shards_info = []
        total_tokens = 0
        total_bytes = 0

        for idx, path in enumerate(shard_paths):
            if not os.path.exists(path):
                continue
            size_bytes = os.path.getsize(path)
            num_tokens = size_bytes // dtype_size

            # Compute SHA-256 checksum for data integrity
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            checksum = sha256_hash.hexdigest()

            shards_info.append(
                {
                    "shard": idx,
                    "filename": os.path.basename(path),
                    "tokens": num_tokens,
                    "size_bytes": size_bytes,
                    "offset": total_tokens,
                    "checksum": checksum,
                }
            )
            total_tokens += num_tokens
            total_bytes += size_bytes

        index_data = {
            "total_shards": len(shards_info),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "shards": shards_info,
        }
        return index_data

    @classmethod
    def save_index(cls, shard_paths: List[str], output_path: str, dtype_size: int = 2) -> None:
        index_data = cls.generate_index(shard_paths, dtype_size=dtype_size)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
