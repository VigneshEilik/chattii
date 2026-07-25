from typing import Dict, Type

from engine.transformer.nn.activation import GELU, ReLU, Sigmoid, SiLU, Softmax, Tanh
from engine.transformer.nn.module import Module
from engine.transformer.nn.normalization import LayerNorm, RMSNorm
from engine.transformer.position.registry import PositionRegistry

__all__ = ["ActivationRegistry", "NormalizationRegistry", "PositionRegistry"]


class ActivationRegistry:
    """
    Registry mapping activation identifiers to classes.
    """

    _registry: Dict[str, Type[Module]] = {
        "relu": ReLU,
        "gelu": GELU,
        "silu": SiLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax,
    }

    @classmethod
    def register(cls, name: str, act_cls: Type[Module]) -> None:
        """Registers a custom activation class."""
        cls._registry[name.lower()] = act_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves target activation type class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown activation: '{name}'")
        return cls._registry[key]


class NormalizationRegistry:
    """
    Registry mapping normalization identifiers to classes.
    """

    _registry: Dict[str, Type[Module]] = {
        "layernorm": LayerNorm,
        "rmsnorm": RMSNorm,
    }

    @classmethod
    def register(cls, name: str, norm_cls: Type[Module]) -> None:
        """Registers a custom normalization class."""
        cls._registry[name.lower()] = norm_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves target normalization type class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown normalization: '{name}'")
        return cls._registry[key]
