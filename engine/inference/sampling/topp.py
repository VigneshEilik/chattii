import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("topp")
class TopPFilter(BaseSamplingFilter):
    """Filters logits keeping only the smallest set of tokens whose cumulative probability is >= p."""

    def __init__(self, p: float = 0.9) -> None:
        self.p = p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or self.p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        return logits
