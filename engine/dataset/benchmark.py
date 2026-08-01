import time
from typing import Any, Dict

from engine.dataset.cleaner.pipeline import CleaningPipeline
from engine.dataset.cleaner.unicode import UnicodeCleaner
from engine.dataset.cleaner.whitespace import WhitespaceCleaner
from engine.dataset.config import DatasetPipelineConfig
from engine.dataset.loader.streaming import StreamingPipeline
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.registry.registry import DatasetRegistry


class DatasetBenchmark:
    """Benchmarks loading, cleaning, and streaming throughput for a registered dataset."""

    @staticmethod
    def benchmark(dataset_name: str) -> Dict[str, Any]:
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

        # 1. Benchmark raw loading speed
        loader = DatasetFactory.create_loader(config)
        start = time.perf_counter()
        load_count = 0
        for _ in loader:
            load_count += 1
        load_time = time.perf_counter() - start

        # 2. Benchmark cleaning speed
        loader2 = DatasetFactory.create_loader(config)
        pipeline = CleaningPipeline(cleaners=[UnicodeCleaner(), WhitespaceCleaner()])
        start = time.perf_counter()
        clean_count = 0
        for sample in loader2:
            pipeline.clean(sample)
            clean_count += 1
        clean_time = time.perf_counter() - start

        # 3. Benchmark streaming pipeline speed
        loader3 = DatasetFactory.create_loader(config)
        streaming = StreamingPipeline(
            loader=loader3, cleaners=[UnicodeCleaner(), WhitespaceCleaner()]
        )
        start = time.perf_counter()
        stream_count = 0
        for _ in streaming:
            stream_count += 1
        stream_time = time.perf_counter() - start

        results = {
            "dataset_name": dataset_name,
            "loading": {
                "samples": load_count,
                "time_s": load_time,
                "samples_per_sec": load_count / max(1e-6, load_time),
            },
            "cleaning": {
                "samples": clean_count,
                "time_s": clean_time,
                "samples_per_sec": clean_count / max(1e-6, clean_time),
            },
            "streaming": {
                "samples": stream_count,
                "time_s": stream_time,
                "samples_per_sec": stream_count / max(1e-6, stream_time),
            },
        }

        print(f"\n--- Dataset Benchmark: {dataset_name} ---")
        for phase, data in results.items():
            if isinstance(data, dict):
                print(f"  [{phase}]")
                for k, v in data.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")

        return results
