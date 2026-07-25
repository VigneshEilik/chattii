# ruff: noqa: E501
from typing import Any, Dict, List

import torch.nn as nn


class WeightDecayFilter:
    """Groups model parameters into weight decay and non-decay groups (bias, LayerNorm)."""

    @staticmethod
    def get_parameter_groups(
        model: nn.Module, weight_decay: float
    ) -> List[Dict[str, Any]]:
        """
        Splits parameters into weight-decay candidate list and bias/norm exclusion list.
        """
        decay_params = []
        nodecay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Biases, normalization layers (RMSNorm, LayerNorm) do not get weight decay
            if (
                "bias" in name
                or "norm" in name
                or "ln" in name
                or name.endswith(".weight")
                and param.dim() == 1
            ):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
