import torch.distributed as dist

from engine.distributed.backend.registry import BackendRegistry


@BackendRegistry.register("gloo")
def init_gloo() -> None:
    """Initializes PyTorch distributed process group using GLOO backend."""
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
