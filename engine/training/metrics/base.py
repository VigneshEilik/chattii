import torch


class MetricEvaluator:
    """Base class for implementing custom training and validation metrics."""

    def evaluate(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> float:
        """Computes metric value based on logits and target class outputs."""
        raise NotImplementedError
