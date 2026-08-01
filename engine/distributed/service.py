from typing import Any, Optional, Tuple

import torch.nn as nn

import engine.distributed.backend.gloo  # noqa: F401
import engine.distributed.backend.mpi  # noqa: F401

# Register backends automatically
import engine.distributed.backend.nccl  # noqa: F401

# Register strategies automatically
import engine.distributed.strategy.cpu  # noqa: F401
import engine.distributed.strategy.ddp  # noqa: F401
import engine.distributed.strategy.deepspeed  # noqa: F401
import engine.distributed.strategy.fsdp  # noqa: F401
import engine.distributed.strategy.single_gpu  # noqa: F401
from engine.distributed.backend.registry import BackendRegistry
from engine.distributed.config import DistributedConfig
from engine.distributed.launcher.environment import DistributedEnvironment
from engine.distributed.strategy.factory import StrategyFactory


class DistributedService:
    """Entry point coordinating process initialization, model sharding, and scale optimizations."""

    def __init__(self, config: Optional[DistributedConfig] = None) -> None:
        self.config = config or DistributedConfig.load_from_yaml()
        self.strategy = None

    def initialize_process_group(self) -> None:
        env = DistributedEnvironment.get_env()
        # Initialize torch distributed group
        if int(env["WORLD_SIZE"]) > 1 or self.config.world_size > 1:
            init_fn = BackendRegistry.get(self.config.backend)
            init_fn()

    def wrap_training_elements(
        self, model: nn.Module, optimizer: Any, dataloader: Any
    ) -> Tuple[nn.Module, Any, Any]:
        self.strategy = StrategyFactory.create(self.config.strategy)
        wrapped_model = self.strategy.wrap_model(model)
        wrapped_opt = self.strategy.wrap_optimizer(optimizer)
        wrapped_loader = self.strategy.wrap_dataloader(dataloader)

        return wrapped_model, wrapped_opt, wrapped_loader
