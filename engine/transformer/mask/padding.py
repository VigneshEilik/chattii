from typing import Optional, Union

import torch

from engine.transformer.math.precision import PrecisionManager


class PaddingMask:
    """
    Generates key-padding attention masks from token sequence tensors.
    """

    @staticmethod
    def create(
        input_ids: torch.Tensor,
        pad_id: int = 0,
        dtype: Optional[Union[str, torch.dtype]] = None,
        additive: bool = True,
    ) -> torch.Tensor:
        """
        Creates padding mask tensor for batch sequences.
        """
        # Boolean mask: True where input_ids == pad_id
        mask = input_ids == pad_id

        if additive:
            target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32
            additive_mask = torch.zeros(
                (input_ids.shape[0], 1, 1, input_ids.shape[1]),
                device=input_ids.device,
                dtype=target_dtype,
            )
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)
            additive_mask.masked_fill_(expanded_mask, float("-inf"))
            return additive_mask

        return mask
