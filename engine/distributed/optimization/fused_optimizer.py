from typing import Any, Dict


class FusedOptimizerWrapper:
    """Configures optimizers to use fused step updates when available."""

    @staticmethod
    def get_fused_kwargs(optimizer_name: str) -> Dict[str, Any]:
        # AdamW and SGD support fused updates in PyTorch 2.0+
        if optimizer_name.lower() in ["adamw", "sgd"]:
            return {"fused": True}
        return {}
