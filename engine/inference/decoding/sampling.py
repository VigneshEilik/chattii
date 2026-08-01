import torch


class SamplingDecoder:
    """Samples from the token probability distribution using multinomial selection."""

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        # Mask out -inf values prior to softmax
        probs = torch.softmax(logits, dim=-1)
        # Avoid issues with zero probabilities
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
