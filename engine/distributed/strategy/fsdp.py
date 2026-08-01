# ruff: noqa: N817
import os
from typing import Any

import torch
import torch.nn as nn

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:
    FSDP = None


@StrategyRegistry.register("fsdp")
class FSDPStrategy(BaseStrategy):
    """Fully Sharded Data Parallel strategy wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if FSDP is not None and torch.cuda.is_available():
            return FSDP(model)
        return model
