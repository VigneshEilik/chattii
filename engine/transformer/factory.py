from typing import Any

from engine.transformer.nn.module import Module
from engine.transformer.position.factory import PositionFactory
from engine.transformer.registry import ActivationRegistry, NormalizationRegistry

__all__ = ["ActivationFactory", "NormalizationFactory", "PositionFactory"]


class ActivationFactory:
    """
    Factory for instantiating activation layers dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """Creates an activation layer instance."""
        cls_type = ActivationRegistry.get(name)
        return cls_type(**kwargs)


class NormalizationFactory:
    """
    Factory for instantiating normalization layers dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """Creates a normalization layer instance."""
        cls_type = NormalizationRegistry.get(name)
        return cls_type(**kwargs)
