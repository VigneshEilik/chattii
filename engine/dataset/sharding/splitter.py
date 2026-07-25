import random
from typing import Any, List, Tuple


class DatasetSplitter:
    """Splits a list of files, paths, or records into train, validation, and test sets."""

    def __init__(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        # Allow slight rounding errors
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

    def split(self, items: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        # Clone list and shuffle deterministically
        shuffled = list(items)
        random.seed(self.seed)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_set = shuffled[:n_train]
        val_set = shuffled[n_train : n_train + n_val]
        test_set = shuffled[n_train + n_val :]

        return train_set, val_set, test_set
