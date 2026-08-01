import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("eta")
class EtaFilter(BaseSamplingFilter):
    """Filters logits keeping only tokens whose probability is >= eta * entropy."""

    def __init__(self, eta: float = 0.002) -> None:
        self.eta = eta

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.eta <= 0.0 or self.eta >= 1.0:
            return logits

        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
        min_threshold = self.eta * entropy

        indices_to_remove = probs < min_threshold
        logits[indices_to_remove] = float("-inf")
        return logits
