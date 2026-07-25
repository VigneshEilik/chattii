from typing import Iterable, Union

import torch


class GradientClipper:
    """Clips parameter gradients by norm or value boundary thresholds."""

    @staticmethod
    def clip_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
    ) -> float:
        """
        Clips parameter gradients in-place by scaling when total norm exceeds max_norm.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        params = [p for p in parameters if p.grad is not None]
        if not params:
            return 0.0

        # Calculate global norm of all gradients combined
        total_norm = 0.0
        for p in params:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in params:
                p.grad.data.mul_(clip_coef)

        return total_norm

    @staticmethod
    def clip_grad_value(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], clip_value: float
    ) -> None:
        """
        Clips parameter gradients in-place to fit inside [-clip_value, clip_value].
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for p in parameters:
            if p.grad is not None:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)
