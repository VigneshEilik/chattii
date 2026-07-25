from typing import Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager


class CausalMask:
    """
    Generates upper-triangular causal attention mask matrices.
    """

    @staticmethod
    def create(
        seq_len: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        additive: bool = True,
    ) -> torch.Tensor:
        """
        Creates a causal mask of shape (seq_len, seq_len).
        If additive=True, masked values are float("-inf") and visible values are 0.0.
        """
        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        # Upper triangular boolean mask (True for masked positions)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=target_device, dtype=torch.bool),
            diagonal=1,
        )

        if additive:
            additive_mask = torch.zeros(
                (seq_len, seq_len), device=target_device, dtype=target_dtype
            )
            additive_mask.masked_fill_(mask, float("-inf"))
            return additive_mask

        return mask
