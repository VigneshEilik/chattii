# ruff: noqa: E501
import os
from typing import Any

from engine.training.callbacks.base import Callback
from engine.training.training_state import TrainingState


class CheckpointCallback(Callback):
    """Callback managing step-based, epoch-based, and best-metric checkpoint selections."""

    def __init__(
        self, save_every_steps: int, checkpoint_dir: str, keep_top_n: int = 3
    ) -> None:
        self.save_every_steps = save_every_steps
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_n = keep_top_n

    def on_step_end(self, state: TrainingState, **kwargs: Any) -> None:
        """Saves dynamic checkpoints after global step thresholds."""
        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # 1. Interval step checkpoint
        if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
            step_path = os.path.join(self.checkpoint_dir, f"step_{state.global_step}")
            trainer.save_checkpoint(step_path)

        # 2. Best loss checkpoint update
        if state.current_loss < state.best_loss:
            state.best_loss = state.current_loss
            best_path = os.path.join(self.checkpoint_dir, "best_loss")
            trainer.save_checkpoint(best_path)

    def on_epoch_end(self, state: TrainingState, **kwargs: Any) -> None:
        """Saves checkpoint at the end of each training epoch."""
        trainer = kwargs.get("trainer")
        if trainer is not None:
            epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{state.epoch}")
            trainer.save_checkpoint(epoch_path)
            # Link/copy to latest
            latest_path = os.path.join(self.checkpoint_dir, "latest")
            trainer.save_checkpoint(latest_path)
