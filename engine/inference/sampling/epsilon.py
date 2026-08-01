import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("epsilon")
class EpsilonFilter(BaseSamplingFilter):
    """Filters logits keeping only tokens whose probability is >= epsilon."""

    def __init__(self, epsilon: float = 0.003) -> None:
        self.epsilon = epsilon

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.epsilon >= 1.0:
            return logits

        probs = torch.softmax(logits, dim=-1)
        indices_to_remove = probs < self.epsilon
        logits[indices_to_remove] = float("-inf")
        return logits
