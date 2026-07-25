import math
from typing import Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module


class SinusoidalPositionalEncoding(Module):
    """
    Sinusoidal Positional Encoding from the original Transformer architecture.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        pe = torch.zeros(max_len, d_model, device=target_device, dtype=target_dtype)
        position = torch.arange(
            0, max_len, dtype=torch.float32, device=target_device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=target_device)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input embeddings tensor.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype, device=x.device)
