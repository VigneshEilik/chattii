from typing import Any

from engine.transformer.model_registry import ModelRegistry
from engine.transformer.nn.module import Module


class ModelFactory:
    """
    Factory for instantiating model architectures dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """
        Creates a model instance matching designated string name.
        """
        cls_type = ModelRegistry.get(name)
        return cls_type(**kwargs)
