from typing import Any, Optional

from engine.training.callbacks.base import Callback
from engine.training.training_state import TrainingState


class LoggingCallback(Callback):
    """Coordinates logging steps across CSV, TensorBoard, and stdout streams."""

    def __init__(
        self,
        log_every_steps: int = 50,
        csv_logger: Optional[Any] = None,
        tb_logger: Optional[Any] = None,
    ) -> None:
        self.log_every_steps = log_every_steps
        self.csv_logger = csv_logger
        self.tb_logger = tb_logger

    def on_step_end(self, state: TrainingState, **kwargs: Any) -> None:
        """Publishes metrics at log step intervals."""
        if state.global_step > 0 and state.global_step % self.log_every_steps == 0:
            metrics = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": state.current_loss,
                "lr": state.learning_rate,
                "tokens_seen": state.tokens_seen,
            }
            metrics.update(state.metrics)

            if self.csv_logger is not None:
                self.csv_logger.log(metrics)

            if self.tb_logger is not None:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tb_logger.add_scalar(
                            f"train/{name}", value, state.global_step
                        )
