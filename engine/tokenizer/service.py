import inspect
from typing import Any, Dict, List, Optional

from engine.tokenizer.base.tokenizer import BaseTokenizer
from engine.tokenizer.factory import TokenizerFactory


class TokenizerService:
    """
    Service orchestration layer serving as the primary API for all tokenizer actions.
    """

    def __init__(self, algorithm: str, **kwargs: Any) -> None:
        self.tokenizer: BaseTokenizer = TokenizerFactory.create(algorithm, **kwargs)

    def train(self, corpus_path: str, loader: Optional[Any] = None) -> None:
        """
        Orchestrates tokenizer training, dynamically passing loader if accepted.
        """
        sig = inspect.signature(self.tokenizer.train)
        if "loader" in sig.parameters:
            self.tokenizer.train(corpus_path, loader=loader)
        else:
            self.tokenizer.train(corpus_path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Transforms text strings to integer ID arrays.
        """
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int]) -> str:
        """
        Reconstructs text from ID arrays.
        """
        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """
        Serializes and saves the active tokenizer configurations.
        """
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        """
        Loads vocabulary configuration mappings.
        """
        self.tokenizer.load(path)

    def get_vocab_size(self) -> int:
        """
        Returns the vocabulary size of the active tokenizer.
        """
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> Dict[str, str]:
        """
        Returns special tokens metadata dictionary.
        """
        return self.tokenizer.get_special_tokens()
