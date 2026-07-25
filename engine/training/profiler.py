from typing import Any, Dict

import torch


class TrainerProfiler:
    """Profiles host RAM and device CUDA VRAM memory usage during training runs."""

    @staticmethod
    def profile_memory() -> Dict[str, Any]:
        """
        Gathers memory allocations statistics from psutil (RAM) and PyTorch (VRAM).
        """
        metrics = {
            "ram_allocated_mb": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_allocated_mb": 0.0,
        }

        # Retrieve system RSS memory footprint
        try:
            import psutil

            process = psutil.Process()
            metrics["ram_allocated_mb"] = round(process.memory_info().rss / 1e6, 2)
        except ImportError:
            pass

        # Retrieve CUDA device VRAM allocations
        if torch.cuda.is_available():
            metrics["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
            metrics["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 2)
            metrics["gpu_peak_allocated_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 2
            )

        return metrics
