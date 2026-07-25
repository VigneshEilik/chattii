from typing import Any

import torch


class LossScaler:
    """Manages loss scaling dynamically to prevent FP16 underflow/overflow."""

    def __init__(
        self,
        init_scale: float = 65536.0,
        scale_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0

        # Try to use PyTorch native GradScaler for CUDA if available
        self.native_scaler = None
        if torch.cuda.is_available():
            try:
                self.native_scaler = torch.amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=scale_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                )
            except Exception:
                pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss value to keep gradients from underflowing."""
        if self.native_scaler is not None:
            return self.native_scaler.scale(loss)
        return loss * self.scale

    def step(self, optimizer: Any) -> bool:
        """
        Unscales gradients, steps optimizer, and updates scale factor.
        Returns False if gradient overflow (NaN/Inf) was detected and step was skipped.
        """
        if self.native_scaler is not None:
            try:
                self.native_scaler.step(optimizer)
                self.native_scaler.update()
                return True
            except RuntimeError as e:
                # If NaN/Inf gradients caused native scaler to skip step
                if "NaN" in str(e) or "overflow" in str(e):
                    self.native_scaler.update()
                    return False
                raise e

        # Custom fallback unscaling and overflow checks
        has_overflow = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # Unscale gradients back to real values
                    p.grad.data.div_(self.scale)
                    if torch.isnan(p.grad.data).any() or torch.isinf(p.grad.data).any():
                        has_overflow = True

        if has_overflow:
            # Backoff scaling factor and skip this optimization iteration
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self.growth_tracker = 0
            optimizer.zero_grad()
            return False

        optimizer.step()

        # Scale growth updates
        self.growth_tracker += 1
        if self.growth_tracker >= self.growth_interval:
            self.scale *= self.scale_factor
            self.growth_tracker = 0

        return True
