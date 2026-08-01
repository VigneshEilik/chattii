import torch
import torch.nn as nn


class DistributedModelCompiler:
    """Wraps model in torch.compile configurations to run graph optimizations."""

    @staticmethod
    def compile(model: nn.Module, mode: str = "default") -> nn.Module:
        if hasattr(torch, "compile"):
            try:
                # Compile model under requested mode
                return torch.compile(model, mode=mode)
            except Exception as e:
                print(
                    f"Compilation warning: Failed to run torch.compile ({e}). Fallback to standard execution."
                )
                return model
        return model
