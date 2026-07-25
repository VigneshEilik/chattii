from typing import Any, Dict


class TrainingState:
    """
    Standard state tracking container capturing epoch, step, loss, and learning rate
    metrics across the entire training lifecycle.
    """

    def __init__(self) -> None:
        self.epoch: int = 0
        self.global_step: int = 0
        self.tokens_seen: int = 0
        self.examples_seen: int = 0

        self.current_loss: float = 0.0
        self.best_loss: float = float("inf")
        self.best_perplexity: float = float("inf")
        self.learning_rate: float = 0.0

        # Dictionary tracking custom metrics (e.g. accuracy, grad_norm, elapsed_time)
        self.metrics: Dict[str, Any] = {}

    def update(self, loss: float, lr: float, tokens: int, examples: int) -> None:
        """Updates primary iteration metrics."""
        self.current_loss = loss
        self.learning_rate = lr
        self.tokens_seen += tokens
        self.examples_seen += examples

    def state_dict(self) -> Dict[str, Any]:
        """Returns state metrics for serialization."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "examples_seen": self.examples_seen,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "best_perplexity": self.best_perplexity,
            "learning_rate": self.learning_rate,
            "metrics": self.metrics,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads state metrics from deserialized dictionary."""
        self.epoch = state_dict.get("epoch", 0)
        self.global_step = state_dict.get("global_step", 0)
        self.tokens_seen = state_dict.get("tokens_seen", 0)
        self.examples_seen = state_dict.get("examples_seen", 0)
        self.current_loss = state_dict.get("current_loss", 0.0)
        self.best_loss = state_dict.get("best_loss", float("inf"))
        self.best_perplexity = state_dict.get("best_perplexity", float("inf"))
        self.learning_rate = state_dict.get("learning_rate", 0.0)
        self.metrics = state_dict.get("metrics", {})
