import time
import tracemalloc
from typing import Any, Callable, Dict


class TokenizerProfiler:
    """
    Profiles RAM consumption, CPU execution times, and peak resource footprints.
    """

    @staticmethod
    def profile_action(
        func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Profiles the target function execution parameters.
        """
        tracemalloc.start()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end_time - start_time
        return {
            "elapsed_seconds": round(elapsed, 6),
            "ram_peak_bytes": peak,
            "ram_peak_mb": round(peak / (1024 * 1024), 4),
            "result": result,
        }
