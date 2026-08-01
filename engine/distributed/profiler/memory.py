from typing import Any, Dict


class MemoryProfiler:
    """Profiles system RAM and peak device memory utilization."""

    @staticmethod
    def get_summary() -> Dict[str, Any]:
        import tracemalloc

        tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "current_ram_mb": current / (1024**2),
            "peak_ram_mb": peak / (1024**2),
        }
