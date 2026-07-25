from typing import Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module
from engine.transformer.nn.parameters import Parameter


class TokenEmbedding(Module):
    """
    Custom Token Embedding lookup layer implemented from scratch.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_id: Optional[int] = 0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_id = pad_id

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        # Standard normal weight initialization scaled by 1 / sqrt(hidden_size)
        w_data = torch.randn(
            (vocab_size, hidden_size), device=target_device, dtype=target_dtype
        ) / (hidden_size**0.5)

        self.register_parameter("weight", Parameter(w_data))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieves embedding vectors for token input IDs.
        """
        clipped_ids = input_ids.clone()
        invalid_mask = (clipped_ids < 0) | (clipped_ids >= self.vocab_size)
        if invalid_mask.any():
            fallback_id = self.pad_id if self.pad_id is not None else 0
            clipped_ids[invalid_mask] = fallback_id

        embeddings = self.weight.data[clipped_ids]

        if self.pad_id is not None:
            pad_mask = (input_ids == self.pad_id).unsqueeze(-1)
            embeddings = embeddings.masked_fill(pad_mask, 0.0)

        return embeddings
