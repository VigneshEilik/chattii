from typing import Any

import torch


class DistributedGradScaler:
    """Manages scaling of gradients to avoid underflow in mixed precision."""

    def __init__(self, enabled: bool = True) -> None:
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=enabled)
            if torch.cuda.is_available()
            else None
        )

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: Any) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
