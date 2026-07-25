import math
from typing import Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.matrix import MatrixOps
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module
from engine.transformer.nn.parameters import Parameter


class Linear(Module):
    """
    Custom Linear Transformation Layer performing Y = X W^T + b from scratch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        # Weight tensor shape: (out_features, in_features)
        w_data = torch.empty(
            (out_features, in_features), device=target_device, dtype=target_dtype
        )
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        torch.nn.init.uniform_(w_data, -bound, bound)

        self.register_parameter("weight", Parameter(w_data))

        if bias:
            b_data = torch.empty(
                (out_features,), device=target_device, dtype=target_dtype
            )
            torch.nn.init.uniform_(b_data, -bound, bound)
            self.register_parameter("bias", Parameter(b_data))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication x @ weight^T + bias.
        """
        output = MatrixOps.matmul(x, MatrixOps.transpose(self.weight.data, -2, -1))
        if self.bias is not None:
            output = output + self.bias.data
        return output
