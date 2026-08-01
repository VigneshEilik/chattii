# ruff: noqa: N817
import os
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry


@StrategyRegistry.register("ddp")
class DDPStrategy(BaseStrategy):
    """Distributed Data Parallel strategy wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        import torch.distributed as dist

        if dist.is_initialized():
            if torch.cuda.is_available():
                return DDP(
                    model, device_ids=[self.local_rank], output_device=self.local_rank
                )
            else:
                return DDP(model)
        return model
