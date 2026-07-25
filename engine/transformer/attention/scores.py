import math
from typing import Optional

import torch

from engine.transformer.math.matrix import MatrixOps
from engine.transformer.nn.module import Module


class AttentionScores(Module):
    """
    Computes scaled dot-product attention scores with optional masking.
    """

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates score matrices of shape (batch_size, num_heads, q_len, kv_len).
        """
        scale = 1.0 / math.sqrt(q.shape[-1])
        # q @ k^T
        scores = MatrixOps.matmul(q, MatrixOps.transpose(k, -2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        return scores
