from torch.optim import Optimizer


class ConstantLRScheduler:
    """Keeps learning rate constant throughout training."""

    def __init__(self, optimizer: Optimizer, base_lr: float) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr

    def step(self, global_step: int) -> float:
        """Sets the optimizer learning rates and returns current LR."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr
        return self.base_lr
