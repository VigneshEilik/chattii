import torch.distributed as dist

from engine.distributed.backend.registry import BackendRegistry


@BackendRegistry.register("nccl")
def init_nccl() -> None:
    """Initializes PyTorch distributed process group using NCCL backend."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
