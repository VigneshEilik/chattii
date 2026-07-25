import os
from typing import List, Union

import numpy as np


class MemoryMappedDataset:
    """Provides a virtually unified, contiguous indexable array over multiple sharded binary files using np.memmap."""

    def __init__(self, shard_paths: List[str], dtype: str = "uint16") -> None:
        self.shard_paths = shard_paths
        self.dtype = np.dtype(dtype)
        self.mmaps = []
        self.sizes = []
        self.offsets = []

        total_size = 0
        for path in shard_paths:
            if not os.path.exists(path):
                continue
            file_size_bytes = os.path.getsize(path)
            size = file_size_bytes // self.dtype.itemsize
            if size == 0:
                continue

            m = np.memmap(path, dtype=self.dtype, mode="r")
            self.mmaps.append(m)
            self.sizes.append(size)
            self.offsets.append(total_size)
            total_size += size

        self.total_size = total_size

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: Union[int, slice]) -> Union[int, np.ndarray]:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total_size)
            res = []
            for i in range(start, stop, step or 1):
                res.append(self._get_single(i))
            return np.array(res, dtype=self.dtype)
        return self._get_single(idx)

    def _get_single(self, idx: int) -> int:
        if idx < 0 or idx >= self.total_size:
            raise IndexError("MemoryMappedDataset index out of range")

        # Binary search or scan to find which shard holds the index
        for i, offset in enumerate(self.offsets):
            if idx < offset + self.sizes[i]:
                return int(self.mmaps[i][idx - offset])

        raise IndexError("MemoryMappedDataset index out of range")

    def close(self) -> None:
        """Closes all underlying memory maps to release file handles."""
        for m in self.mmaps:
            if hasattr(m, "_mmap") and m._mmap is not None:
                m._mmap.close()
        self.mmaps.clear()
