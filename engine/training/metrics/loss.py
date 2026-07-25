import torch

from engine.training.loss.cross_entropy import CrossEntropyLoss
from engine.training.metrics.base import MetricEvaluator


class LossMetric(MetricEvaluator):
    """Metric wrapper that computes custom cross-entropy loss."""

    def __init__(self) -> None:
        self.loss_fn = CrossEntropyLoss()

    def evaluate(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> float:
        """Computes current cross-entropy loss."""
        self.loss_fn.ignore_index = ignore_index
        return float(self.loss_fn(logits, targets).item())
