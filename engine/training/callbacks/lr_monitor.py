from typing import Any

from engine.training.callbacks.base import Callback
from engine.training.training_state import TrainingState


class LRMonitorCallback(Callback):
    """Monitors learning rate changes and records them to the training state."""

    def on_step_start(self, state: TrainingState, **kwargs: Any) -> None:
        """Captures active learning rate before executing gradients step."""
        trainer = kwargs.get("trainer")
        if trainer is not None and trainer.optimizer is not None:
            active_lr = trainer.optimizer.param_groups[0]["lr"]
            state.learning_rate = active_lr
