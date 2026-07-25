import time
from typing import Any, Dict

from engine.dataset.config import DatasetPipelineConfig
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.registry.registry import DatasetRegistry


class DatasetProfiler:
    """Profiles CPU time and peak memory estimates during dataset streaming operations."""

    @staticmethod
    def profile(dataset_name: str) -> Dict[str, Any]:
        registry = DatasetRegistry()
        entry = registry.get(dataset_name)
        if not entry:
            print(f"Dataset '{dataset_name}' not found in registry.")
            return {}

        config = DatasetPipelineConfig(
            name=entry.get("name", dataset_name),
            path=entry.get("path", ""),
            loader_type=entry.get("format", "txt"),
        )
        loader = DatasetFactory.create_loader(config)

        import tracemalloc
        tracemalloc.start()

        start = time.perf_counter()
        sample_count = 0
        total_chars = 0
        for sample in loader:
            sample_count += 1
            total_chars += len(sample.text)

        elapsed = time.perf_counter() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results = {
            "dataset_name": dataset_name,
            "samples_streamed": sample_count,
            "total_characters": total_chars,
            "elapsed_seconds": elapsed,
            "samples_per_second": sample_count / max(1e-6, elapsed),
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

        print(f"\n--- Dataset Profile: {dataset_name} ---")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        return results
