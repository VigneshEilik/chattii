from typing import Iterator, Optional, Tuple

import torch

from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.batching.collator import DataCollator
from engine.dataset.batching.prefetch import PrefetchBuffer


class ChatiiDataLoader:
    """Consumes TokenizedSamples and yields collated PyTorch tensor batches (with prefetching)."""

    def __init__(
        self,
        dataset_stream: Iterator[TokenizedSample],
        batch_size: int = 16,
        collator: Optional[DataCollator] = None,
        prefetch_batches: int = 4,
    ) -> None:
        self.dataset_stream = dataset_stream
        self.batch_size = batch_size
        self.collator = collator or DataCollator()
        self.prefetch_batches = prefetch_batches

    def stream(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        batch = []
        for sample in self.dataset_stream:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collator.collate(batch)
                batch = []

        if batch:
            yield self.collator.collate(batch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        iterator = self.stream()
        if self.prefetch_batches > 0:
            return PrefetchBuffer(iterator, buffer_size=self.prefetch_batches)
        return iterator
