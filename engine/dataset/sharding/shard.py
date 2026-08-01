import os
from typing import Iterator, List

from engine.dataset.base.dataset import TextSample


class DatasetSharder:
    """Splits a stream of text samples into sharded files on disk."""

    def __init__(
        self, out_dir: str, prefix: str = "shard", samples_per_shard: int = 1000
    ) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.samples_per_shard = samples_per_shard
        os.makedirs(out_dir, exist_ok=True)

    def shard_stream(self, stream: Iterator[TextSample]) -> List[str]:
        shard_paths = []
        shard_idx = 0
        buffer = []

        def save_buffer_to_shard():
            nonlocal shard_idx
            if not buffer:
                return
            path = os.path.join(self.out_dir, f"{self.prefix}_{shard_idx:05d}.txt")
            with open(path, "w", encoding="utf-8") as f:
                for item in buffer:
                    f.write(item.text.replace("\n", " ") + "\n")
            shard_paths.append(path)
            shard_idx += 1
            buffer.clear()

        for sample in stream:
            buffer.append(sample)
            if len(buffer) >= self.samples_per_shard:
                save_buffer_to_shard()

        save_buffer_to_shard()
        return shard_paths
