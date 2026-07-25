from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseTrainer(ABC):
    """
    Stable abstract base interface for all Chatii tokenizer training algorithms.
    """

    @abstractmethod
    def train(
        self, corpus_path: str, vocab_size: int, special_tokens: List[str]
    ) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        """
        Trains on the raw corpus path and builds vocabulary mapping rules.
        Returns:
            Tuple[Dict[str, int], List[Tuple[str, str]]]: (vocab_mapping, merges_list)
        """
        pass
