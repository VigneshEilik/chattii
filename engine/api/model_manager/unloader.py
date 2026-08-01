import gc

import torch

from engine.inference.service import InferenceService


class ModelUnloader:
    """Safely unloads weights from memory registries and triggers garbage collection."""

    def unload(self, service: InferenceService) -> None:
        service.clear_cache()
        # Nullify model references
        service.model = None
        service.tokenizer = None
        service.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
