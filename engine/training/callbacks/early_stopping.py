from typing import Any

from engine.training.callbacks.base import Callback
from engine.training.exceptions import EarlyStoppingTrigger
from engine.training.training_state import TrainingState


class EarlyStoppingCallback(Callback):
    """Monitors validation loss to trigger early halting on training plateau."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.bad_runs: int = 0

    def on_validation_end(self, state: TrainingState, **kwargs: Any) -> None:
        """Triggers training halt if validation metrics cease to improve."""
        val_loss = state.metrics.get("val_loss")
        if val_loss is None:
            return

        # Check for loss improvement exceeding delta threshold
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_runs = 0
        else:
            self.bad_runs += 1
            if self.bad_runs >= self.patience:
                raise EarlyStoppingTrigger(
                    f"Early stopping triggered: validation loss failed to improve "
                    f"for {self.patience} validation steps."
                )
        state.best_loss = min(state.best_loss, self.best_loss)
