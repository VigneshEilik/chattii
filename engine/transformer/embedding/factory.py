from typing import Any

from engine.transformer.embedding.registry import EmbeddingRegistry
from engine.transformer.nn.module import Module


class EmbeddingFactory:
    """
    Factory for instantiating embedding layers dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """
        Creates an embedding layer instance.
        """
        cls_type = EmbeddingRegistry.get(name)
        return cls_type(**kwargs)
