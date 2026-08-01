from typing import Any, Dict

from engine.distributed.planner.memory_estimator import ModelMemoryEstimator


class BatchSizeEstimator:
    """Finds the maximum batch size that fits in a target VRAM budget."""

    @staticmethod
    def estimate_optimal_batch_size(
        num_params: int,
        gpu_memory_gb: float,
        seq_len: int,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        precision: str = "bf16",
    ) -> Dict[str, Any]:
        # Target safety margin: use at most 80% of VRAM
        budget_gb = gpu_memory_gb * 0.8

        optimal_batch = 1
        for batch in range(1, 1024):
            estimates = ModelMemoryEstimator.estimate(
                num_params=num_params,
                batch_size=batch,
                seq_len=seq_len,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                precision=precision,
            )
            if estimates["total_estimated_gb"] > budget_gb:
                break
            optimal_batch = batch

        return {
            "optimal_batch_size": optimal_batch,
            "estimated_memory_gb": estimates["total_estimated_gb"],
            "vram_limit_gb": gpu_memory_gb,
        }
