import torch.nn as nn


class DDPPlaceholder:
    """Placeholder stub for future DistributedDataParallel (DDP) integration."""

    @staticmethod
    def wrap_model(model: nn.Module) -> nn.Module:
        """Passes model through unmodified for single-device execution."""
        return model
