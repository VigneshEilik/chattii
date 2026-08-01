import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("typical")
class TypicalFilter(BaseSamplingFilter):
    """Filters logits based on token typicality relative to current conditional entropy."""

    def __init__(self, mass: float = 0.9) -> None:
        self.mass = mass

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.mass <= 0.0 or self.mass >= 1.0:
            return logits

        # Compute typical sampling filter mask
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
        shifted_log_probs = torch.abs(-torch.log(probs + 1e-10) - entropy)

        sorted_shifted, sorted_indices = torch.sort(shifted_log_probs, dim=-1)
        sorted_probs = probs.gather(-1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.mass
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        return logits
