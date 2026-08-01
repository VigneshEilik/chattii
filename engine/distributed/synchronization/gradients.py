from typing import List

import torch
import torch.distributed as dist


class GradientSynchronizer:
    """Manually aggregates and averages gradients across all active ranks."""

    @staticmethod
    def synchronize(parameters: List[torch.nn.Parameter]) -> None:
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        for param in parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
