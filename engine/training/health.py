# ruff: noqa: C901
from typing import Any, Dict, List

import torch
import torch.nn as nn


class TrainerHealthChecker:
    """Audits gradient statistics and loss curves to detect diverging/failing runs."""

    @staticmethod
    def check_health(model: nn.Module, loss_history: List[float]) -> Dict[str, Any]:
        """
        Validates gradient values, checks for NaNs, and analyzes loss curve stability.
        """
        issues = []

        # 1. Audit loss history trends
        if len(loss_history) >= 2:
            if loss_history[-1] > loss_history[0] * 1.5:
                issues.append(
                    "Divergence alert: Training loss is increasing significantly."
                )

        # 2. Audit parameter gradient flows
        has_gradients = False
        zero_gradients = []
        nan_gradients = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_gradients = True
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_gradients.append(name)
                    if param.grad.abs().sum().item() == 0.0:
                        zero_gradients.append(name)

        if not has_gradients:
            issues.append("Gradients alert: Model has no parameter gradients computed.")
        if nan_gradients:
            issues.append(
                f"NaNs alert: Parameters contain NaN gradients: {nan_gradients[:3]}"
            )
        if len(zero_gradients) == sum(1 for p in model.parameters() if p.requires_grad):
            issues.append(
                "Dead weights alert: All active parameters have zero gradients."
            )

        return {"healthy": len(issues) == 0, "issues": issues}
