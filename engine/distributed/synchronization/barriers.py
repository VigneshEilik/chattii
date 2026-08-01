import torch.distributed as dist


class DistributedBarrier:
    """Synchronizes processes across the distributed group."""

    @staticmethod
    def wait() -> None:
        if dist.is_initialized():
            dist.barrier()
