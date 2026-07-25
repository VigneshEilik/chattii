from typing import List, Tuple

import torch

from engine.transformer.nn.module import Module


class FeedForwardValidator:
    """
    Validates output shapes and numeric stability of FFN blocks.
    """

    @staticmethod
    def validate_feedforward(
        layer: Module, x: torch.Tensor, out: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        Runs shape compliance and NaN/Inf checks on FFN forward outputs.
        """
        errors = []
        if out.shape != x.shape:
            errors.append(
                f"FFN output shape {tuple(out.shape)} does not match "
                f"input shape {tuple(x.shape)}"
            )
        if torch.isnan(out).any():
            errors.append("NaN values detected in FFN output tensor.")
        if torch.isinf(out).any():
            errors.append("Inf values detected in FFN output tensor.")
        return len(errors) == 0, errors
