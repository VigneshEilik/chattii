# ruff: noqa: E501
from typing import Any, Dict


class MemoryEstimator:
    """Estimates training parameter, gradient, optimizer, and activation memory footprints."""

    @staticmethod
    def estimate(
        num_parameters: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        num_layers: int,
        optimizer_name: str = "adamw",
        precision: str = "float32",
    ) -> Dict[str, Any]:
        """
        Computes static memory usage for parameters, grads, optimizer variables,
        and provides an approximation of activation memory in Megabytes.
        """
        # Bytes per parameter value depending on precision
        bytes_per_val = 4 if precision == "float32" else 2

        # 1. Parameter and Gradient memory
        param_mem = num_parameters * bytes_per_val
        grad_mem = num_parameters * bytes_per_val

        # 2. Optimizer state memory (AdamW stores 2 fp32 moments; SGD stores 1 moment)
        if optimizer_name.lower() == "adamw":
            opt_mem = num_parameters * 4 * 2
        elif optimizer_name.lower() == "sgd":
            opt_mem = num_parameters * 4
        else:
            opt_mem = 0

        # 3. Activation memory (approximate standard self-attention + MLP activation layer size)
        act_mem = num_layers * batch_size * seq_len * d_model * 20 * 4

        total_bytes = param_mem + grad_mem + opt_mem + act_mem

        return {
            "parameter_mb": round(param_mem / 1e6, 2),
            "gradient_mb": round(grad_mem / 1e6, 2),
            "optimizer_mb": round(opt_mem / 1e6, 2),
            "activation_mb": round(act_mem / 1e6, 2),
            "total_mb": round(total_bytes / 1e6, 2),
        }
