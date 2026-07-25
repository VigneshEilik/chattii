import torch.nn as nn


class FSDPPlaceholder:
    """Placeholder stub for future FullyShardedDataParallel (FSDP) integration."""

    @staticmethod
    def wrap_model(model: nn.Module) -> nn.Module:
        """Passes model through unmodified for single-device execution."""
        return model
