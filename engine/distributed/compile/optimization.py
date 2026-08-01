from typing import Any, Dict


class CompilerOptimizationSettings:
    """Manages compiler optimizations and flag configurations."""

    @staticmethod
    def get_defaults() -> Dict[str, Any]:
        return {
            "triton_enabled": True,
            "epilogue_fusion": True,
            "dynamic_shapes": False,
        }
