import torch.distributed as dist

from engine.distributed.backend.registry import BackendRegistry


@BackendRegistry.register("mpi")
def init_mpi() -> None:
    """Initializes PyTorch distributed process group using MPI backend."""
    if not dist.is_initialized():
        dist.init_process_group(backend="mpi")
