from typing import Any, Dict


class ModelMemoryEstimator:
    """Estimates parameters, gradients, optimizer state, and activation memory requirements."""

    @staticmethod
    def estimate(
        num_params: int,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        precision: str = "bf16",
    ) -> Dict[str, Any]:
        bytes_per_val = 2 if precision in ["fp16", "bf16"] else 4

        # Model params and gradients
        param_mem = num_params * bytes_per_val
        grad_mem = num_params * bytes_per_val

        # AdamW uses float32 master weights (4 bytes) + 1st moment (4 bytes) + 2nd moment (4 bytes) = 12 bytes/param
        opt_mem = num_params * 12

        # KV cache memory
        head_dim = hidden_dim // num_heads
        kv_cache_mem = (
            2 * num_layers * num_heads * head_dim * seq_len * batch_size * 2
        )  # assume FP16/BF16 cache

        # Activation memory estimate (rough rule of thumb: 34 * batch * seq * layers * hidden)
        activation_mem = 34 * batch_size * seq_len * num_layers * hidden_dim

        total_bytes = param_mem + grad_mem + opt_mem + kv_cache_mem + activation_mem

        return {
            "parameters_gb": param_mem / (1024**3),
            "gradients_gb": grad_mem / (1024**3),
            "optimizer_gb": opt_mem / (1024**3),
            "kv_cache_gb": kv_cache_mem / (1024**3),
            "activations_gb": activation_mem / (1024**3),
            "total_estimated_gb": total_bytes / (1024**3),
        }
