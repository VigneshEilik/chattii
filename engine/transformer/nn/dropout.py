import torch

from engine.transformer.nn.module import Module


class Dropout(Module):
    """
    Inverted Dropout layer respecting training/evaluation module state.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability p must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies dropout during training; acts as an identity function in eval mode.
        """
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(dtype=x.dtype)
        return x * mask / (1.0 - self.p)
