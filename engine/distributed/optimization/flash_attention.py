from typing import Optional

import torch


class FlashAttentionHook:
    """Provides hooks and interface fallbacks for external fast attention kernels."""

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        backend: str = "auto",
    ) -> torch.Tensor:
        # Fallback to PyTorch's native scaled dot product attention (which auto-resolves to flash attention if available)
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
