from typing import Any

from engine.tokenizer.base.tokenizer import BaseTokenizer
from engine.tokenizer.registry import TokenizerRegistry


class TokenizerFactory:
    """
    Factory interface to construct registered tokenizer algorithm instances.
    """

    @staticmethod
    def create(algorithm: str, **kwargs: Any) -> BaseTokenizer:
        """
        Creates a concrete tokenizer instance mapped by its algorithm name.
        """
        tokenizer_cls = TokenizerRegistry.get(algorithm)
        return tokenizer_cls(**kwargs)
