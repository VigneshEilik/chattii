from typing import Any

from torch.optim import Optimizer

from engine.training.scheduler.registry import SchedulerRegistry
from engine.training.scheduler.warmup import WarmupSchedulerWrapper


class SchedulerFactory:
    """Factory for instantiating learning rate schedulers with optional warmup."""

    @staticmethod
    def create(
        name: str,
        optimizer: Optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """
        Creates scheduler, calculates decay window steps after warmup,
        and wraps with WarmupSchedulerWrapper if warmup_steps > 0.
        """
        cls_type = SchedulerRegistry.get(name)

        # Calculate steps remaining for actual decay after warmup is completed
        decay_steps = max(1, total_steps - warmup_steps)

        if name == "constant":
            base_scheduler = cls_type(optimizer, base_lr)
        else:
            base_scheduler = cls_type(
                optimizer, base_lr, total_steps=decay_steps, min_lr=min_lr, **kwargs
            )

        if warmup_steps > 0:
            return WarmupSchedulerWrapper(
                optimizer, base_scheduler, warmup_steps=warmup_steps
            )

        return base_scheduler
