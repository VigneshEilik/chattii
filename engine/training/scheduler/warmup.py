from typing import Any

from torch.optim import Optimizer


class WarmupSchedulerWrapper:
    """Wraps a decay scheduler to apply a linear warmup phase first."""

    def __init__(
        self, optimizer: Optimizer, base_scheduler: Any, warmup_steps: int
    ) -> None:
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.base_lr = base_scheduler.base_lr

    def step(self, global_step: int) -> float:
        """Sets warmup learning rate or delegates to the decay scheduler."""
        if global_step < self.warmup_steps:
            # Linear scaling from 0.0 to base_lr over warmup steps
            lr = (global_step / max(1, self.warmup_steps)) * self.base_lr
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return lr
        else:
            # Shift step index to start decay right after warmup ends
            return self.base_scheduler.step(global_step - self.warmup_steps)
