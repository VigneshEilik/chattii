import math
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


class GradientMonitor:
    """Calculates statistics (L2 norms, min, max, NaNs) over parameter gradients."""

    @staticmethod
    def monitor(parameters: Iterable[nn.Parameter]) -> Dict[str, Any]:
        """
        Aggregates gradient norm, range, and stability checks.
        """
        grads = [p.grad.data for p in parameters if p.grad is not None]
        if not grads:
            return {
                "norm": 0.0,
                "max": 0.0,
                "min": 0.0,
                "has_nan": False,
                "has_inf": False,
            }

        total_norm = 0.0
        max_val = float("-inf")
        min_val = float("inf")
        has_nan = False
        has_inf = False

        for g in grads:
            total_norm += g.norm(2).item() ** 2
            max_val = max(max_val, g.max().item())
            min_val = min(min_val, g.min().item())

            if torch.isnan(g).any():
                has_nan = True
            if torch.isinf(g).any():
                has_inf = True

        return {
            "norm": math.sqrt(total_norm),
            "max": max_val,
            "min": min_val,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
