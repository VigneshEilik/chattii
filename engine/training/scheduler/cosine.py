import math

from torch.optim import Optimizer


class CosineDecayLRScheduler:
    """Decays learning rate using a cosine curve down to min_lr over total_steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr

    def step(self, global_step: int) -> float:
        """Computes next step cosine decayed learning rate."""
        if global_step >= self.total_steps:
            lr = self.min_lr
        else:
            decay_ratio = global_step / self.total_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.base_lr - self.min_lr)

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
