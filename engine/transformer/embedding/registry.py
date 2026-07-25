from typing import Dict, Type

from engine.transformer.embedding.lm_head import LMHead
from engine.transformer.embedding.token import TokenEmbedding
from engine.transformer.nn.module import Module


class EmbeddingRegistry:
    """
    Registry for Chatii embedding-layer implementations.
    """

    _registry: Dict[str, Type[Module]] = {
        "token": TokenEmbedding,
        "lm_head": LMHead,
    }

    @classmethod
    def register(cls, name: str, embedding_cls: Type[Module]) -> None:
        """Registers a custom embedding implementation."""
        cls._registry[name.lower()] = embedding_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves target embedding type class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(
                f"Unknown embedding type '{name}'. "
                f"Available options: {list(cls._registry.keys())}"
            )
        return cls._registry[key]
