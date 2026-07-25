import random
from typing import Iterator, List


class BatchSampler:
    """Divides dataset indices into batch lists, with optional shuffling."""

    def __init__(self, num_samples: int, batch_size: int, shuffle: bool = True) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch = indices[i : i + self.batch_size]
            yield batch

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size
