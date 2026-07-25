from typing import Dict, Type

from engine.training.trainer import BaseTrainer, SFTTrainer


class TrainerRegistry:
    """Registry mapping trainer names to trainer classes."""

    _registry: Dict[str, Type[BaseTrainer]] = {
        "sft": SFTTrainer,
    }

    @classmethod
    def register(cls, name: str, trainer_cls: Type[BaseTrainer]) -> None:
        """Registers a custom training strategy class."""
        cls._registry[name.lower().strip()] = trainer_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTrainer]:
        """Resolves target trainer class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown training strategy: '{name}'")
        return cls._registry[key]
