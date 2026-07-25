import time
from typing import Any, Dict

import torch

from engine.transformer.embedding.token import TokenEmbedding


class EmbeddingBenchmark:
    """
    Measures lookup speed, throughput, and latency of token embeddings.
    """

    @staticmethod
    def run_benchmark(
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        batch_size: int = 4,
        seq_len: int = 1024,
        iterations: int = 200,
    ) -> Dict[str, Any]:
        """
        Benchmarks TokenEmbedding lookup operations on synthetic input sequences.
        """
        layer = TokenEmbedding(vocab_size, hidden_size)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

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
