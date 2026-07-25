import torch

from engine.training.metrics.base import MetricEvaluator


class AccuracyMetric(MetricEvaluator):
    """Metric wrapper that computes token-level next-token prediction accuracy."""

    def evaluate(
        self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> float:
        """
        Calculates ratio of correctly predicted tokens to total non-padding tokens.
        """
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)

        mask = flat_targets != ignore_index
        if not mask.any():
            return 0.0

        active_logits = flat_logits[mask]
        active_targets = flat_targets[mask]

        preds = torch.argmax(active_logits, dim=-1)
        correct = (preds == active_targets).sum().item()
        return float(correct / active_targets.size(0))
