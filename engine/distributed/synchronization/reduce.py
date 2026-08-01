import torch
import torch.distributed as dist


class DistributedReduce:
    """Reduces tensors from all ranks into a single target rank."""

    @staticmethod
    def reduce(
        tensor: torch.Tensor, dst: int = 0, op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        if dist.is_initialized():
            dist.reduce(tensor, dst=dst, op=op)
        return tensor
