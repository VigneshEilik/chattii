from typing import Iterator

from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.binary.mmap import MemoryMappedDataset


class BinaryDatasetReader:
    """Reads sequences of fixed block sizes from a MemoryMappedDataset and yields TokenizedSamples."""

    def __init__(self, mmap_dataset: MemoryMappedDataset, block_size: int = 2048) -> None:
        self.mmap_dataset = mmap_dataset
        self.block_size = block_size

    def stream(self) -> Iterator[TokenizedSample]:
        total_tokens = len(self.mmap_dataset)
        idx = 0
        while idx + self.block_size <= total_tokens:
            chunk = self.mmap_dataset[idx : idx + self.block_size]
            yield TokenizedSample(
                input_ids=chunk.tolist(),
                sample_id=f"bin_{idx}",
                attention_mask=[1] * self.block_size,
                labels=chunk.tolist(),
                metadata={"offset": idx},
            )
            idx += self.block_size

    def __iter__(self) -> Iterator[TokenizedSample]:
        return self.stream()
