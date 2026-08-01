import time
from typing import Any, Dict


class ThroughputProfiler:
    """Measures token processing throughput and scaling metrics."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def get_throughput(self, total_tokens: int) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.start
        return {
            "tokens_per_second": total_tokens / max(1e-6, elapsed),
            "elapsed_seconds": elapsed,
        }
