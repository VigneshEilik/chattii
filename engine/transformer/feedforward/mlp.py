from typing import Optional, Union

import torch

from engine.transformer.nn.activation import GELU
from engine.transformer.nn.linear import Linear
from engine.transformer.nn.module import Module


class MLP(Module):
    """
    Standard Multi-Layer Perceptron (Linear -> GELU -> Linear).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.act = GELU()
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        self.register_module("w1", self.w1)
        self.register_module("act", self.act)
        self.register_module("w2", self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes FFN forward pass.
        """
        return self.w2(self.act(self.w1(x)))
