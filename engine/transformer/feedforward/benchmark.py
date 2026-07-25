import time
from typing import Any, Dict

import torch

from engine.transformer.feedforward.factory import FeedForwardFactory


class FeedForwardBenchmark:
    """
    Benchmarks FeedForward (MLP / SwiGLU) layer throughput and latencies.
    """

    @staticmethod
    def run_benchmark(
        name: str = "swiglu",
        d_model: int = 512,
        d_ff: int = 2048,
        batch_size: int = 4,
        seq_len: int = 1024,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Runs feedforward layer benchmarks.
        """
        layer = FeedForwardFactory.create(name, d_model=d_model, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        # Warmup
        for _ in range(10):
            _ = layer(x)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer(x)
        end = time.perf_counter()

        elapsed = end - start
        total_tokens = iterations * batch_size * seq_len
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        return {
            "iterations": iterations,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_tokens_per_sec": round(tokens_per_sec, 2),
            "latency_ms_per_iter": round((elapsed / iterations) * 1000, 4),
        }
