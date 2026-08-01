import time
from typing import Any, Dict, List, Optional

from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


class InferenceBenchmarkSuite:
    """Audits generation latency and throughput across prompt sizes (32, 128, 512, 2048, 4096 tokens)."""

    def __init__(self, service: InferenceService) -> None:
        self.service = service

    def run_benchmark(
        self, prompt_lengths: Optional[List[int]] = None, max_new_tokens: int = 64
    ) -> List[Dict[str, Any]]:
        if prompt_lengths is None:
            prompt_lengths = [32, 128, 512]
        if not self.service.model:
            self.service.load_model()

        results = []
        for plen in prompt_lengths:
            # Generate dummy prompt of length plen
            dummy_prompt = "a" * plen

            config = GenerationConfig(config_dict={"max_new_tokens": max_new_tokens})
            start_time = time.perf_counter()
            response = self.service.generate(dummy_prompt, config=config)
            latency = time.perf_counter() - start_time
            tps = response.generated_tokens / latency if latency > 0 else 0.0

            results.append(
                {
                    "prompt_length": plen,
                    "generation_length": response.generated_tokens,
                    "latency_seconds": latency,
                    "tps": tps,
                }
            )
        return results
