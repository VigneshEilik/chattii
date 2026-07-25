import torch

from engine.training.metrics.base import MetricEvaluator
from engine.training.metrics.loss import LossMetric
from engine.training.validation.perplexity import PerplexityEvaluator


class PerplexityMetric(MetricEvaluator):
    """Metric wrapper that computes language model perplexity."""

    def __init__(self) -> None:
        self.loss_metric = LossMetric()

    def evaluate(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> float:
        """Computes current perplexity value."""
        loss = self.loss_metric.evaluate(logits, targets, ignore_index)
        return PerplexityEvaluator.compute(loss)
