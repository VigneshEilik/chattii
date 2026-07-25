import math

import torch

from engine.transformer.nn.module import Module


class ReLU(Module):
    """Rectified Linear Unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


class GELU(Module):
    """Gaussian Error Linear Unit activation with tanh approximation or exact erf."""

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "tanh":
            return (
                0.5
                * x
                * (
                    1.0
                    + torch.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SiLU(Module):
    """Sigmoid Linear Unit (Swish-1) activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Sigmoid(Module):
    """Sigmoid activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Tanh(Module):
    """Hyperbolic tangent activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Softmax(Module):
    """Softmax activation with max subtraction for numerical stability."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(x, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x - max_val)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
