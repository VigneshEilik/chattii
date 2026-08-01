import torch
import torch.distributed as dist


class DistributedAllReduce:
    """Aggregates tensors from all nodes and shares the results with all ranks."""

    @staticmethod
    def all_reduce(
        tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor
