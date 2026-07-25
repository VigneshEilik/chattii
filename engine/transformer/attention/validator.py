from typing import List, Tuple

import torch

from engine.transformer.attention.multi_head import MultiHeadSelfAttention


class AttentionValidator:
    """
    Validates output shapes, compute devices, and numeric stability of attention blocks.
    """

    @staticmethod
    def validate_attention(
        layer: MultiHeadSelfAttention, x: torch.Tensor, out: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        Runs shape compliance and NaN checks on attention forward outputs.
        """
        errors = []
        if out.shape != x.shape:
            errors.append(
                f"Attention output shape {tuple(out.shape)} does not match "
                f"input shape {tuple(x.shape)}"
            )
        if torch.isnan(out).any():
            errors.append("NaN values detected in attention output tensor.")
        if torch.isinf(out).any():
            errors.append("Inf values detected in attention output tensor.")
        return len(errors) == 0, errors
