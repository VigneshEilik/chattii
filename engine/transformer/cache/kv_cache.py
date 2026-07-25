from typing import Optional, Tuple

import torch


class KVCache:
    """
    Manages Key-Value states for a single Attention layer during inference decoding.
    """

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key-value projection tensors along the sequence dimension (dim 2).
        Shapes of input k, v: (batch_size, num_heads, new_seq_len, head_dim).
        """
        if self.k is None or self.v is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reset(self) -> None:
        """
        Clears the cached Key and Value states.
        """
        self.k = None
        self.v = None
