import time
from typing import Any, Dict

import torch

from engine.transformer.math.matrix import MatrixOps
from engine.transformer.model import ChatiiModel


class MathBenchmark:
    """
    Benchmarks primitive tensor mathematical operations.
    """

    @staticmethod
    def run_benchmark(iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmarks custom matmul throughput and latency.
        """
        x = torch.randn(4, 128, 512)
        w = torch.randn(512, 512)

        # Warmup
        for _ in range(10):
            _ = MatrixOps.matmul(x, w.t())

        start = time.perf_counter()
        for _ in range(iterations):
            _ = MatrixOps.matmul(x, w.t())
        end = time.perf_counter()

        elapsed = end - start
        total_tokens = iterations * 4 * 128
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        return {
            "iterations": iterations,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_tokens_per_sec": round(tokens_per_sec, 2),
            "latency_ms_per_iter": round((elapsed / iterations) * 1000, 4),
        }


class TransformerBenchmark:
    """
    Benchmarks token processing speed and forward pass throughput of ChatiiModel.
    """

    @staticmethod
    def run_benchmark(
        model: ChatiiModel,
        batch_size: int = 2,
        seq_len: int = 128,
        iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Executes model speed benchmark iterations.
        """
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        # Warmup iterations
        for _ in range(3):
            _ = model(x)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
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
