from typing import Any

import torch.nn as nn

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry


@StrategyRegistry.register("cpu")
class CPUStrategy(BaseStrategy):
    """CPU fallback execution strategy."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model.cpu()
