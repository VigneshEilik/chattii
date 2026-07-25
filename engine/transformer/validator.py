from typing import List, Tuple

import torch

from engine.transformer.model import ChatiiModel

__all__ = ["MathValidator", "TransformerValidator"]


class MathValidator:
    """
    Validation engine for primitive mathematical operations.
    """

    @staticmethod
    def validate_output(x: torch.Tensor, name: str) -> Tuple[bool, List[str]]:
        """
        Validates stability (NaN/Inf) of a tensor.
        """
        errors = []
        if torch.isnan(x).any():
            errors.append(f"NaN values detected in {name} output.")
        if torch.isinf(x).any():
            errors.append(f"Inf values detected in {name} output.")
        return len(errors) == 0, errors


class TransformerValidator:
    """
    Validation engine verifying weight tying and numerical health.
    """

    @staticmethod
    def validate_model(
        model: ChatiiModel, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        Validates model output shapes and parameter count.
        """
        errors = []

        # 1. Output Dimension check
        b, t = input_ids.shape
        expected_shape = (b, t, model.config.vocab_size)
        if logits.shape != expected_shape:
            errors.append(
                f"Output logits shape {tuple(logits.shape)} does not match "
                f"expected {expected_shape}."
            )

        # 2. NaN/Inf numerical checks
        if torch.isnan(logits).any():
            errors.append("NaN values detected in model logits output.")
        if torch.isinf(logits).any():
            errors.append("Inf values detected in model logits output.")

        # 3. Weight tying verification
        if model.weight_tying:
            if model.token_embedding.weight.data is not model.lm_head.weight.data:
                errors.append(
                    "Model specifies weight tying, but weights point to "
                    "different memory locations."
                )

        # 4. Parameter quantity checks
        if model.num_parameters() <= 0:
            errors.append("Model parameter count is zero or negative.")

        return len(errors) == 0, errors
