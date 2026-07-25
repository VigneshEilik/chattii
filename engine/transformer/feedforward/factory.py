from typing import Any

from engine.transformer.feedforward.registry import FeedForwardRegistry
from engine.transformer.nn.module import Module


class FeedForwardFactory:
    """
    Factory for instantiating feedforward layers dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """
        Creates a feedforward network layer instance.
        """
        cls_type = FeedForwardRegistry.get(name)
        return cls_type(**kwargs)
