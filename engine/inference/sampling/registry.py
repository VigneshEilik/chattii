from typing import Callable, Dict, Type

import torch


class BaseSamplingFilter:
    """Base interface for logits filter strategies."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SamplingRegistry:
    """Registry class managing all logits sampling filters."""

    _registry: Dict[str, Type[BaseSamplingFilter]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Type[BaseSamplingFilter]], Type[BaseSamplingFilter]]:
        def decorator(
            filter_class: Type[BaseSamplingFilter],
        ) -> Type[BaseSamplingFilter]:
            cls._registry[name] = filter_class
            return filter_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseSamplingFilter]:
        if name not in cls._registry:
            raise KeyError(f"Sampling filter '{name}' is not registered.")
        return cls._registry[name]
