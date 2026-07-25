from typing import Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.matrix import MatrixOps
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module
from engine.transformer.nn.parameters import Parameter


class LMHead(Module):
    """
    Language Modeling Head layer projecting hidden states to vocabulary logits.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        bias: bool = False,
        tied_weight: Optional[Parameter] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        if tied_weight is not None:
            self.weight = tied_weight
        else:
            w_data = torch.randn(
                (vocab_size, hidden_size), device=target_device, dtype=target_dtype
            ) / (hidden_size**0.5)
            self.register_parameter("weight", Parameter(w_data))

        if bias:
            b_data = torch.zeros(
                (vocab_size,), device=target_device, dtype=target_dtype
            )
            self.register_parameter("bias", Parameter(b_data))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden states onto the vocabulary token logit dimensions.
        """
        logits = MatrixOps.matmul(x, MatrixOps.transpose(self.weight.data, -2, -1))
        if self.bias is not None:
            logits = logits + self.bias.data
        return logits
