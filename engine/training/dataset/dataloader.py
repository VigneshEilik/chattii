# ruff: noqa: E501
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from engine.training.dataset.dataset import ChatiiDataset
from engine.training.dataset.sequence import SequenceShifter


class ChatiiDataLoader:
    """
    DataLoader wrapper that batches sequences and shifts inputs/targets for next-token SFT.
    """

    def __init__(
        self,
        dataset: ChatiiDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Pin memory to speed up CUDA transfers if available
        pin_memory = torch.cuda.is_available()

        self.loader = TorchDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.loader:
            # batch is of shape (B, seq_len + 1)
            inputs, targets = SequenceShifter.shift(batch)
            yield inputs, targets

    def __len__(self) -> int:
        return len(self.loader)
