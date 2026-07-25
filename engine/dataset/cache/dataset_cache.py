import os
from typing import List, Optional

import numpy as np


class DatasetCache:
    """Caches preprocessed token IDs to disk to bypass repetitive tokenization."""

    def __init__(self, cache_dir: str = ".cache/dataset") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, fingerprint: str) -> str:
        return os.path.join(self.cache_dir, f"{fingerprint}.bin")

    def get(self, fingerprint: str, dtype: str = "uint16") -> Optional[np.ndarray]:
        path = self.get_cache_path(fingerprint)
        if os.path.exists(path):
            try:
                return np.fromfile(path, dtype=np.dtype(dtype))
            except Exception:
                return None
        return None

    def set(self, fingerprint: str, token_ids: List[int], dtype: str = "uint16") -> str:
        path = self.get_cache_path(fingerprint)
        arr = np.array(token_ids, dtype=np.dtype(dtype))
        arr.tofile(path)
        return path
