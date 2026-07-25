from typing import Optional, Union

import torch

from engine.transformer.attention.multi_head import MultiHeadSelfAttention
from engine.transformer.cache.kv_cache import KVCache
from engine.transformer.factory import NormalizationFactory
from engine.transformer.feedforward.factory import FeedForwardFactory
from engine.transformer.nn.module import Module


class DecoderBlock(Module):
    """
    Single Decoder block with Pre-Normalization and Residual connections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        activation: str = "swiglu",
        normalization: str = "rmsnorm",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.attn_norm = NormalizationFactory.create(
            normalization, dim=d_model, device=device, dtype=dtype
        )
        self.ffn_norm = NormalizationFactory.create(
            normalization, dim=d_model, device=device, dtype=dtype
        )

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ffn = FeedForwardFactory.create(
            activation, d_model=d_model, d_ff=d_ff, device=device, dtype=dtype
        )

        self.register_module("attn_norm", self.attn_norm)
        self.register_module("ffn_norm", self.ffn_norm)
        self.register_module("attn", self.attn)
        self.register_module("ffn", self.ffn)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Executes pre-normalization, multi-head attention, FFN, and residual additions.
        """
        # Pre-Norm Attention
        norm_x = self.attn_norm(x)
        attn_out = self.attn(norm_x, mask=mask, cache=cache, use_cache=use_cache)
        x = x + attn_out

        # Pre-Norm FFN
        norm_ffn_x = self.ffn_norm(x)
        ffn_out = self.ffn(norm_ffn_x)
        x = x + ffn_out

        return x
