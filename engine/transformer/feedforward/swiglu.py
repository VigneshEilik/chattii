from typing import Optional, Union

import torch

from engine.transformer.nn.activation import SiLU
from engine.transformer.nn.linear import Linear
from engine.transformer.nn.module import Module


class SwiGLU(Module):
    """
    SwiGLU Feed-Forward Network layer (SiLU(xW_gate) * xW_up) * W_down.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.w_gate = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w_up = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.act = SiLU()

        self.register_module("w_gate", self.w_gate)
        self.register_module("w_up", self.w_up)
        self.register_module("w_down", self.w_down)
        self.register_module("act", self.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes SwiGLU forward pass.
        """
        gate_out = self.act(self.w_gate(x))
        up_out = self.w_up(x)
        return self.w_down(gate_out * up_out)
