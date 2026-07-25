from typing import Dict, Type

import torch.nn as nn

from engine.training.loss.cross_entropy import CrossEntropyLoss


class LossRegistry:
    """Registry mapping loss names to their corresponding nn.Module classes."""

    _registry: Dict[str, Type[nn.Module]] = {
        "cross_entropy": CrossEntropyLoss,
    }

    @classmethod
    def register(cls, name: str, loss_cls: Type[nn.Module]) -> None:
        """Registers a custom loss class."""
        cls._registry[name.lower().strip()] = loss_cls

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Resolves target loss class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown loss type: '{name}'")
        return cls._registry[key]
