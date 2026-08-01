import time
from typing import Any, Dict

import torch

from engine.inference.service import InferenceService


class InferenceProfiler:
    """Profiles TTFT, tokens/sec, and memory footprint of Chatii model."""

    def __init__(self, service: InferenceService) -> None:
        self.service = service

    def profile_request(self, prompt: str, max_new_tokens: int = 64) -> Dict[str, Any]:
        if not self.service.model:
            raise RuntimeError("Service model must be loaded prior to profiling.")

        start_time = time.perf_counter()

        # Enable GPU memory profiling if available
        vram_start = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated()

        # Generate response
        from engine.inference.config import GenerationConfig

        config = GenerationConfig(config_dict={"max_new_tokens": max_new_tokens})
        response = self.service.generate(prompt, config=config)

        vram_peak = 0
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() - vram_start

        latency = time.perf_counter() - start_time
        tps = response.generated_tokens / latency if latency > 0 else 0.0

        return {
            "prompt_length": response.prompt_tokens,
            "generated_length": response.generated_tokens,
            "latency": latency,
            "ttft": response.latency_seconds
            / max(1, response.generated_tokens),  # Approximation
            "tps": tps,
            "peak_vram_bytes": vram_peak,
        }
