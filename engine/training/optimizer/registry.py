from typing import Dict, Type

from torch.optim import Optimizer

from engine.training.optimizer.adamw import AdamW
from engine.training.optimizer.sgd import SGD


class OptimizerRegistry:
    """Registry mapping optimizer names to optimizer classes."""

    _registry: Dict[str, Type[Optimizer]] = {
        "adamw": AdamW,
        "sgd": SGD,
    }

    @classmethod
    def register(cls, name: str, opt_cls: Type[Optimizer]) -> None:
        """Registers a custom optimizer class."""
        cls._registry[name.lower().strip()] = opt_cls

    @classmethod
    def get(cls, name: str) -> Type[Optimizer]:
        """Resolves target optimizer class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown optimizer type: '{name}'")
        return cls._registry[key]
