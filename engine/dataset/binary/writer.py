import os
from typing import Iterator, List

import numpy as np

from engine.dataset.base.dataset import TokenizedSample


class BinaryDatasetWriter:
    """Serializes streams of token IDs to sharded binary files for high-speed loading."""

    def __init__(self, out_dir: str, prefix: str = "dataset", dtype: str = "uint16") -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.dtype = np.dtype(dtype)
        os.makedirs(out_dir, exist_ok=True)

    def write(
        self, samples: Iterator[TokenizedSample], tokens_per_shard: int = 1_000_000
    ) -> List[str]:
        shard_paths = []
        shard_idx = 0
        buffer = []

        for sample in samples:
            buffer.extend(sample.input_ids)
            while len(buffer) >= tokens_per_shard:
                to_write = buffer[:tokens_per_shard]
                buffer = buffer[tokens_per_shard:]

                path = os.path.join(self.out_dir, f"{self.prefix}_{shard_idx:05d}.bin")
                np.array(to_write, dtype=self.dtype).tofile(path)
                shard_paths.append(path)
                shard_idx += 1

        if buffer:
            path = os.path.join(self.out_dir, f"{self.prefix}_{shard_idx:05d}.bin")
            np.array(buffer, dtype=self.dtype).tofile(path)
            shard_paths.append(path)

        return shard_paths
