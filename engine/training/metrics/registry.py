from typing import Dict, Type

from engine.training.metrics.accuracy import AccuracyMetric
from engine.training.metrics.base import MetricEvaluator
from engine.training.metrics.loss import LossMetric
from engine.training.metrics.perplexity import PerplexityMetric


class MetricsRegistry:
    """Registry mapping metric names to their corresponding MetricEvaluator classes."""

    _registry: Dict[str, Type[MetricEvaluator]] = {
        "loss": LossMetric,
        "accuracy": AccuracyMetric,
        "perplexity": PerplexityMetric,
    }

    @classmethod
    def register(cls, name: str, metric_cls: Type[MetricEvaluator]) -> None:
        """Registers a custom metric class."""
        cls._registry[name.lower().strip()] = metric_cls

    @classmethod
    def get(cls, name: str) -> Type[MetricEvaluator]:
        """Resolves target metric class."""
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(f"Unknown metric type: '{name}'")
        return cls._registry[key]
