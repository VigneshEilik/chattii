from typing import Optional, Tuple, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module
from engine.transformer.nn.parameters import Parameter


class LayerNorm(Module):
    """
    Standard Layer Normalization implemented from scratch.
    """

    def __init__(
        self,
        normalized_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        dim: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        shape = normalized_shape if normalized_shape is not None else dim
        if shape is None:
            raise ValueError(
                "Either normalized_shape or dim must be provided to LayerNorm."
            )

        if isinstance(shape, int):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        if self.elementwise_affine:
            w_data = torch.ones(
                self.normalized_shape, device=target_device, dtype=target_dtype
            )
            b_data = torch.zeros(
                self.normalized_shape, device=target_device, dtype=target_dtype
            )
            self.register_parameter("weight", Parameter(w_data))
            self.register_parameter("bias", Parameter(b_data))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(var + self.eps)
        if (
            self.elementwise_affine
            and self.weight is not None
            and self.bias is not None
        ):
            normalized = normalized * self.weight.data + self.bias.data
        return normalized


class RMSNorm(Module):
    """
    Root Mean Square Normalization (RMSNorm) used in modern LLMs (LLaMA, Gemini).
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        normalized_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        target_dim = dim
        if target_dim is None and normalized_shape is not None:
            if isinstance(normalized_shape, tuple):
                target_dim = normalized_shape[0]
            else:
                target_dim = normalized_shape
        if target_dim is None:
            raise ValueError(
                "Either dim or normalized_shape must be provided to RMSNorm."
            )

        self.dim = target_dim
        self.eps = eps

        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        w_data = torch.ones((target_dim,), device=target_device, dtype=target_dtype)
        self.register_parameter("weight", Parameter(w_data))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.data
