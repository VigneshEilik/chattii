import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("min_p")
class MinPFilter(BaseSamplingFilter):
    """Filters logits keeping only tokens whose probability is >= min_p * max_prob."""

    def __init__(self, min_p: float = 0.05) -> None:
        self.min_p = min_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_p <= 0.0 or self.min_p >= 1.0:
            return logits

        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        min_prob_threshold = max_prob * self.min_p

        indices_to_remove = probs < min_prob_threshold
        logits[indices_to_remove] = float("-inf")
        return logits
