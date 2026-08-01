import math
from typing import Any, Dict


class GPURequirementEstimator:
    """Estimates how many GPUs are needed for training based on target memory footprints."""

    @staticmethod
    def estimate_gpus(
        total_memory_gb: float, gpu_capacity_gb: float = 24.0
    ) -> Dict[str, Any]:
        # Leave a 20% memory safety margin
        usable_gpu_mem = gpu_capacity_gb * 0.8
        required_gpus = max(1, math.ceil(total_memory_gb / usable_gpu_mem))

        return {
            "required_gpus": required_gpus,
            "usable_memory_per_gpu_gb": usable_gpu_mem,
            "safety_margin_gb": gpu_capacity_gb * 0.2,
        }
