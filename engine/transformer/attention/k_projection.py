from typing import Optional, Union

import torch

from engine.transformer.nn.linear import Linear
from engine.transformer.nn.module import Module


class KeyProjection(Module):
    """
    Decoupled linear projection layer for Attention Keys.
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
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj = Linear(
            d_model, num_heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.register_module("proj", self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden states to Key dimensions.
        """
        projected = self.proj(x)
        b, t, _ = projected.shape
        return projected.view(b, t, self.num_heads, self.head_dim)
