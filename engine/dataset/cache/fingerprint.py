import hashlib
import json
import os
from typing import Any, Dict, List


class DatasetFingerprint:
    """Calculates SHA-256 hashes of configurations and files to ensure training reproducibility."""

    @staticmethod
    def calculate(config_dict: Dict[str, Any], file_paths: List[str]) -> str:
        sha256_hash = hashlib.sha256()

        # Hash configuration settings
        config_str = json.dumps(config_dict, sort_keys=True)
        sha256_hash.update(config_str.encode("utf-8"))

        # Hash file paths, sizes, and modification times to detect source edits
        for path in sorted(file_paths):
            if os.path.exists(path):
                stat = os.stat(path)
                descriptor = f"{path}_{stat.st_size}_{stat.st_mtime}"
                sha256_hash.update(descriptor.encode("utf-8"))

        return sha256_hash.hexdigest()
