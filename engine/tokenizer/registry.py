from typing import Any, Dict, List, Type

from engine.tokenizer.algorithms.bpe.algorithm import BPETokenizer
from engine.tokenizer.algorithms.character import CharacterTokenizer
from engine.tokenizer.algorithms.word import WordTokenizer
from engine.tokenizer.base.tokenizer import BaseTokenizer


class StubTokenizer(BaseTokenizer):
    """
    Stub class for future algorithm integrations.
    """

    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    def train(self, corpus_path: str, loader: Any = None) -> None:
        raise NotImplementedError(
            f"Tokenizer algorithm '{self.algorithm}' is not implemented yet."
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raise NotImplementedError(
            f"Tokenizer algorithm '{self.algorithm}' is not implemented yet."
        )

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError(
            f"Tokenizer algorithm '{self.algorithm}' is not implemented yet."
        )

    def save(self, path: str) -> None:
        raise NotImplementedError(
            f"Tokenizer algorithm '{self.algorithm}' is not implemented yet."
        )

    def load(self, path: str) -> None:
        raise NotImplementedError(
            f"Tokenizer algorithm '{self.algorithm}' is not implemented yet."
        )

    def get_vocab_size(self) -> int:
        return 0

    def get_special_tokens(self) -> Dict[str, str]:
        return {}


class TokenizerRegistry:
    """
    Dynamic registry holding mappings of algorithm string labels.
    """

    _registry: Dict[str, Type[BaseTokenizer]] = {
        "character": CharacterTokenizer,
        "word": WordTokenizer,
        "bpe": BPETokenizer,
    }

    @classmethod
    def register(cls, name: str, tokenizer_cls: Type[BaseTokenizer]) -> None:
        """
        Registers a new tokenizer class mapping.
        """
        cls._registry[name.lower()] = tokenizer_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTokenizer]:
        """
        Resolves tokenizer class mappings. Returns a dynamic stub if future.
        """
        name_lower = name.lower()
        if name_lower in cls._registry:
            return cls._registry[name_lower]

        stubs = {"sentencepiece", "wordpiece", "unigram", "bbpe"}
        if name_lower in stubs:

            class DynamicStub(StubTokenizer):
                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(name_lower)

            return DynamicStub

        raise ValueError(
            f"Algorithm '{name}' is not registered in the TokenizerRegistry."
        )
