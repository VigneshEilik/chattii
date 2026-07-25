from typing import Dict, Type

from engine.transformer.nn.module import Module
from engine.transformer.position.alibi import ALiBiPositionalBias
from engine.transformer.position.rope import RotaryPositionalEmbedding
from engine.transformer.position.sinusoidal import SinusoidalPositionalEncoding


class PositionRegistry:
    """
    Registry for positional encoding strategies (RoPE, Sinusoidal, ALiBi).
    """

    _registry: Dict[str, Type[Module]] = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "rope": RotaryPositionalEmbedding,
        "alibi": ALiBiPositionalBias,
    }

    @classmethod
    def register(cls, name: str, position_cls: Type[Module]) -> None:
        """Registers a custom positional encoding implementation."""
        cls._registry[name.lower()] = position_cls

    @classmethod
    def get(cls, name: str) -> Type[Module]:
        """Resolves algorithm label to position encoding class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(
                f"Unknown positional encoding '{name}'. "
                f"Available options: {list(cls._registry.keys())}"
            )
        return cls._registry[key]
