from typing import Optional, Union

import torch

from engine.transformer.attention.k_projection import KeyProjection
from engine.transformer.attention.output_projection import OutputProjection
from engine.transformer.attention.q_projection import QueryProjection
from engine.transformer.attention.scores import AttentionScores
from engine.transformer.attention.softmax import AttentionSoftmax
from engine.transformer.attention.v_projection import ValueProjection
from engine.transformer.cache.kv_cache import KVCache
from engine.transformer.nn.dropout import Dropout
from engine.transformer.nn.module import Module
from engine.transformer.position.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(Module):
    """
    Decoupled Multi-Head Self-Attention layer supporting RoPE and KV caching.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = QueryProjection(d_model, num_heads, self.head_dim, device, dtype)
        self.k_proj = KeyProjection(d_model, num_heads, self.head_dim, device, dtype)
        self.v_proj = ValueProjection(d_model, num_heads, self.head_dim, device, dtype)

        self.rope = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len, rope_theta, device
        )

        self.scores_layer = AttentionScores()
        self.softmax_layer = AttentionSoftmax(dim=-1)
        self.dropout_layer = Dropout(dropout)
        self.out_proj = OutputProjection(
            d_model, num_heads, self.head_dim, device, dtype
        )

        self.register_module("q_proj", self.q_proj)
        self.register_module("k_proj", self.k_proj)
        self.register_module("v_proj", self.v_proj)
        self.register_module("rope", self.rope)
        self.register_module("scores_layer", self.scores_layer)
        self.register_module("softmax_layer", self.softmax_layer)
        self.register_module("dropout_layer", self.dropout_layer)
        self.register_module("out_proj", self.out_proj)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Computes attention forward pass with optional KV cache updating.
        """
        b, t, _ = x.shape

        # 1. Project Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 2. Compute RoPE starting position
        start_pos = 0
        if use_cache and cache is not None and cache.k is not None:
            start_pos = cache.k.shape[2]

        # 3. Apply RoPE
        xq, xk = self.rope(xq, xk, start_pos=start_pos)

        # 4. Transpose sequence & heads
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. KV Cache lookup and update
        if use_cache and cache is not None:
            keys, values = cache.update(xk, xv)
        else:
            keys, values = xk, xv

        # 6. Scaled Dot-Product Scores
        scores = self.scores_layer(xq, keys, mask)

        # 7. Softmax & Dropout
        probs = self.softmax_layer(scores)
        probs = self.dropout_layer(probs)

        # 8. Weighted Sum Context
        context = torch.matmul(probs, values)

        # 9. Concatenate heads and project output
        context = context.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(context)
