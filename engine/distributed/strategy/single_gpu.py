from typing import Any

import torch
import torch.nn as nn

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry


@StrategyRegistry.register("single_gpu")
class SingleGPUStrategy(BaseStrategy):
    """Single GPU execution strategy."""

    def __init__(self, device_id: int = 0, **kwargs: Any) -> None:
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.to(self.device)
