import inspect
from typing import Any

from engine.transformer.nn.module import Module
from engine.transformer.position.registry import PositionRegistry


class PositionFactory:
    """
    Factory for instantiating positional encodings dynamically.
    """

    @staticmethod
    def create(name: str, **kwargs: Any) -> Module:
        """Instantiates positional encoding matching label."""
        cls_type = PositionRegistry.get(name)

        dim = kwargs.get("dim") or kwargs.get("d_model")
        max_len = kwargs.get("max_len") or kwargs.get("max_seq_len")

        params = {
            "d_model": dim,
            "dim": dim,
            "max_len": max_len,
            "max_seq_len": max_len,
            "device": kwargs.get("device"),
            "dtype": kwargs.get("dtype"),
        }

        sig = inspect.signature(cls_type.__init__)
        filtered = {k: v for k, v in params.items() if k in sig.parameters}

        for k, v in kwargs.items():
            if k not in params and k in sig.parameters:
                filtered[k] = v

        return cls_type(**filtered)
