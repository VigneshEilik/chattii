import torch

from engine.transformer.nn.activation import Softmax
from engine.transformer.nn.module import Module


class AttentionSoftmax(Module):
    """
    Decoupled numerically stable Softmax layer for attention probability distribution.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.softmax = Softmax(dim=dim)
        self.register_module("softmax", self.softmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies softmax normalization along the designated dim.
        """
        return self.softmax(x)
