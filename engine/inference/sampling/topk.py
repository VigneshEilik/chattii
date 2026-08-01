import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("topk")
class TopKFilter(BaseSamplingFilter):
    """Filters logits keeping only the K highest probability tokens."""

    def __init__(self, k: int = 50) -> None:
        self.k = k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.k <= 0 or self.k >= logits.size(-1):
            return logits

        # Get top K values and indices
        indices_to_remove = logits < torch.topk(logits, self.k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits
