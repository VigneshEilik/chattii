from typing import Optional, Union

import torch

from engine.transformer.mask.causal import CausalMask
from engine.transformer.mask.padding import PaddingMask


class CombinedMask:
    """
    Combines causal and padding mask matrices into a unified attention mask tensor.
    """

    @staticmethod
    def create(
        input_ids: torch.Tensor,
        pad_id: int = 0,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> torch.Tensor:
        """
        Creates combined causal + padding mask of shape (batch, 1, seq_len, seq_len).
        """
        seq_len = input_ids.shape[1]

        causal = CausalMask.create(
            seq_len, device=input_ids.device, dtype=dtype, additive=True
        )
        causal_4d = causal.unsqueeze(0).unsqueeze(0)

        padding_4d = PaddingMask.create(
            input_ids, pad_id=pad_id, dtype=dtype, additive=True
        )

        return torch.minimum(causal_4d, padding_4d)
