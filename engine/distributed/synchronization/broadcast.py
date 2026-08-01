import torch
import torch.distributed as dist


class DistributedBroadcast:
    """Broadcasts a tensor from source rank to all other group processes."""

    @staticmethod
    def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor
