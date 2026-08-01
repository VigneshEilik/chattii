import random
from typing import Any, Iterator, List


class Sampler:
    """Base dataset sampling interface."""

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples dataset indices sequentially (useful for evaluation)."""

    def __init__(self, data_source: Any) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Samples dataset indices randomly (shuffled)."""

    def __init__(self, data_source: Any) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


class WeightedSampler(Sampler):
    """Samples dataset indices with proportional probabilities."""

    def __init__(
        self, weights: List[float], num_samples: int, replacement: bool = True
    ) -> None:
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.weights)))
        choices = random.choices(indices, weights=self.weights, k=self.num_samples)
        return iter(choices)

    def __len__(self) -> int:
        return self.num_samples


class DistributedSampler(Sampler):
    """Subsamples datasets across multiple DDP replicas (distributed training)."""

    def __init__(
        self,
        data_source: Any,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            g = random.Random(self.seed + self.epoch)
            g.shuffle(indices)

        # Slice for rank replica partition
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return (len(self.data_source) + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TemperatureSampler(Sampler):
    """Applies temperature scaling to sample probabilities (useful for mixing multi-domain text datasets)."""

    def __init__(
        self, weights: List[float], num_samples: int, temperature: float = 1.0
    ) -> None:
        self.weights = weights
        self.num_samples = num_samples
        self.temperature = temperature

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.weights)))
        # Apply temperature scaling: prob = w^(1/T)
        scaled_weights = [w ** (1.0 / self.temperature) for w in self.weights]
        choices = random.choices(indices, weights=scaled_weights, k=self.num_samples)
        return iter(choices)

    def __len__(self) -> int:
        return self.num_samples
