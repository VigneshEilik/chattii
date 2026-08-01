from typing import Any

import torch.nn as nn

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry


@StrategyRegistry.register("deepspeed")
class DeepSpeedStrategy(BaseStrategy):
    """DeepSpeed strategy stub supporting Zero redundancy optimizers."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def wrap_optimizer(self, optimizer: Any) -> Any:
        return optimizer
