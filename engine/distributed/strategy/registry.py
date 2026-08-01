from typing import Any, Callable, Dict, Type

import torch.nn as nn


class BaseStrategy:
    """Base interface for all training execution strategies."""

    def wrap_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def wrap_optimizer(self, optimizer: Any) -> Any:
        return optimizer

    def wrap_dataloader(self, dataloader: Any) -> Any:
        return dataloader


class StrategyRegistry:
    """Registry class managing distributed and local strategy modules."""

    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
        def decorator(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
            cls._registry[name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        if name not in cls._registry:
            raise KeyError(f"Strategy '{name}' is not registered.")
        return cls._registry[name]
