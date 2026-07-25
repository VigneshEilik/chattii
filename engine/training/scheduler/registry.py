from typing import Any, Dict, Type

from engine.training.scheduler.constant import ConstantLRScheduler
from engine.training.scheduler.cosine import CosineDecayLRScheduler
from engine.training.scheduler.linear import LinearDecayLRScheduler


class SchedulerRegistry:
    """Registry mapping learning rate scheduler names to their classes."""

    _registry: Dict[str, Type[Any]] = {
        "constant": ConstantLRScheduler,
        "linear": LinearDecayLRScheduler,
        "cosine": CosineDecayLRScheduler,
    }

    @classmethod
    def register(cls, name: str, sched_cls: Type[Any]) -> None:
        """Registers a custom scheduler class."""
        cls._registry[name.lower().strip()] = sched_cls

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        """Resolves target scheduler class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown scheduler type: '{name}'")
        return cls._registry[key]
