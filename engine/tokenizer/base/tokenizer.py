from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTokenizer(ABC):
    """
    Stable abstract base interface for all Chatii tokenizer algorithms.
    All tokenizers must subclass this and implement these methods.
    """

    @abstractmethod
    def train(self, corpus_path: str, loader: Optional[Any] = None) -> None:
        """
        Trains the tokenizer on a raw text corpus file using configuration values.
        """
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Transforms raw string text into a list of integer token IDs.
        """
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Transforms a list of integer token IDs back into a reconstructed string.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serializes and saves vocabulary, merges, and configuration files to a path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Deserializes and loads vocabulary, merges, and configurations from a path.
        """
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Returns the total vocabulary size, including special tokens.
        """
        pass

    @abstractmethod
    def get_special_tokens(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping special token names (e.g., 'pad')
        to their values (e.g., '<pad>').
        """
        pass
