from typing import Any, Dict

import torch


class GPUProfiler:
    """Measures active GPU hardware configurations and temperature characteristics."""

    @staticmethod
    def profile_gpus() -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        return {
            "gpu_available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024**2),
        }
