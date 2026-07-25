from typing import Dict, Type

from engine.transformer.feedforward.mlp import MLP
from engine.transformer.feedforward.swiglu import SwiGLU
from engine.transformer.nn.module import Module


class FeedForwardRegistry:
    """
    Registry mapping feedforward network identifiers to classes.
    """

    _registry: Dict[str, Type[Module]] = {
        "mlp": MLP,
        "gelu": MLP,
        "swiglu": SwiGLU,
        "silu": SwiGLU,
    }

    @classmethod
    def register(cls, name: str, ffn_cls: Type[Module]) -> None:
        """Registers a custom feed-forward layer implementation."""
        cls._registry[name.lower()] = ffn_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves identifier to feed-forward class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(
                f"Unknown feedforward type '{name}'. "
                f"Available options: {list(cls._registry.keys())}"
            )
        return cls._registry[key]
