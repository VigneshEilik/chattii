from typing import Dict, Type

from engine.transformer.model import ChatiiModel
from engine.transformer.nn.module import Module


class ModelRegistry:
    """
    Registry mapping model architectures to their implementation classes.
    """

    _registry: Dict[str, Type[Module]] = {
        "chatii": ChatiiModel,
    }

    @classmethod
    def register(cls, name: str, model_cls: Type[Module]) -> None:
        """Registers a custom model class."""
        cls._registry[name.lower()] = model_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves target model type class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(
                f"Unknown model architecture '{name}'. "
                f"Available options: {list(cls._registry.keys())}"
            )
        return cls._registry[key]
