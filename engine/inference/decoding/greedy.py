import torch


class GreedyDecoder:
    """Selects the highest probability token at each decoding step (argmax)."""

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)
