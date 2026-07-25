import sys
import time
from typing import Any

from engine.training.callbacks.base import Callback
from engine.training.training_state import TrainingState


class ProgressCallback(Callback):
    """Callback that prints a live ASCII progress bar of training metrics to stdout."""

    def __init__(self) -> None:
        self.start_time = 0.0

    def on_train_start(self, state: TrainingState, **kwargs: Any) -> None:
        self.start_time = time.perf_counter()
        print("\n--- Training Run Initiated ---")

    def on_step_end(self, state: TrainingState, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self.start_time
        tokens_per_sec = state.tokens_seen / elapsed if elapsed > 0 else 0.0

        # Build ASCII loss bar (normalized scale from 0 to 10)
        loss = state.current_loss
        bar_len = 10
        filled = min(bar_len, max(0, int(loss)))
        loss_bar = "#" * filled + "-" * (bar_len - filled)

        # Build ASCII LR bar (normalized scale up to 1e-3)
        lr = state.learning_rate
        lr_filled = min(bar_len, max(0, int(lr * 10000)))
        lr_bar = "=" * lr_filled + " " * (bar_len - lr_filled)

        # Print live status update to line
        sys.stdout.write(
            f"\rStep {state.global_step:04d} | Epoch {state.epoch:02d} | "
            f"Loss: {loss:6.4f} [{loss_bar}] | "
            f"LR: {lr:.2e} [{lr_bar}] | "
            f"Speed: {tokens_per_sec:7.1f} tok/s"
        )
        sys.stdout.flush()

    def on_train_end(self, state: TrainingState, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(f"\n--- Training Completed Successfully in {elapsed:.2f} seconds ---")
