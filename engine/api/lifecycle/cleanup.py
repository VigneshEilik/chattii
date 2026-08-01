import gc

import torch

from engine.api.state.state import ApplicationState


def execute_cleanup(state: ApplicationState) -> None:
    """Frees system RAM, temp folders, and triggers CUDA VRAM collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
