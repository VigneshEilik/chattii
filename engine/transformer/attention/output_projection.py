from typing import Optional, Union

import torch

from engine.transformer.nn.linear import Linear
from engine.transformer.nn.module import Module


class OutputProjection(Module):
    """
    Decoupled linear projection mapping combined head outputs back to d_model space.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.proj = Linear(
            num_heads * head_dim, d_model, bias=False, device=device, dtype=dtype
        )
        self.register_module("proj", self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps (B, T, num_heads * head_dim) tensor back to (B, T, d_model).
        """
        return self.proj(x)
