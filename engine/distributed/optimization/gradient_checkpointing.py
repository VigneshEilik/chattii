import torch.nn as nn


class GradientCheckpointingManager:
    """Enables layer-level gradient checkpointing to reduce activation memory footprints."""

    @staticmethod
    def enable(model: nn.Module) -> None:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            # Fallback scan for transformer blocks
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
