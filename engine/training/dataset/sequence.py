from typing import Tuple

import torch


class SequenceShifter:
    """Shifts token lists to construct target tokens for next-token prediction."""

    @staticmethod
    def shift(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits sequence of tokens into input and target variables shifted by 1.
        e.g., tokens = [1, 2, 3, 4] -> inputs = [1, 2, 3], targets = [2, 3, 4]
        """
        if tokens.dim() == 1:
            return tokens[:-1], tokens[1:]
        elif tokens.dim() == 2:
            return tokens[:, :-1], tokens[:, 1:]
        else:
            raise ValueError(
                f"Expected 1D or 2D tensor, got dimension size: {tokens.dim()}"
            )
