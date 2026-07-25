import time
import tracemalloc
from typing import Any, Dict

import torch

from engine.transformer.model import ChatiiModel


class TransformerProfiler:
    """
    Profiles execution latency and memory usage of ChatiiModel forward passes.
    """

    @staticmethod
    def profile_forward(
        model: ChatiiModel, batch_size: int = 2, seq_len: int = 128
    ) -> Dict[str, Any]:
        """
        Profiles a forward pass step, tracking peak memory overhead.
        """
        x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

        tracemalloc.start()
        start = time.perf_counter()

        logits = model(x)

        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end - start
        return {
            "elapsed_seconds": round(elapsed, 6),
            "ram_peak_bytes": peak,
            "ram_peak_mb": round(peak / (1024 * 1024), 4),
            "logits_shape": tuple(logits.shape),
        }
