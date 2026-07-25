from typing import Any

from engine.training.callbacks.base import Callback
from engine.training.training_state import TrainingState


class ValidationCallback(Callback):
    """Callback triggering model evaluations over validation sets at step intervals."""

    def __init__(self, eval_every_steps: int) -> None:
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, state: TrainingState, **kwargs: Any) -> None:
        """Triggers validator run."""
        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        if state.global_step > 0 and state.global_step % self.eval_every_steps == 0:
            trainer.evaluate()
