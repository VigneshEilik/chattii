from typing import Any, Dict

from engine.dataset.config import DatasetPipelineConfig
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.registry.registry import DatasetRegistry


class DatasetInspector:
    """Audits a registered dataset and prints summary metrics to the console."""

    @staticmethod
    def inspect(dataset_name: str) -> Dict[str, Any]:
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

        sample_count = 0
        total_chars = 0
        max_len = 0
        min_len = float("inf")

        for sample in loader:
            sample_count += 1
            length = len(sample.text)
            total_chars += length
            if length > max_len:
                max_len = length
            if length < min_len:
                min_len = length

        avg_len = total_chars / max(1, sample_count)

        results = {
            "dataset_name": dataset_name,
            "path": entry.get("path", ""),
            "format": entry.get("format", "txt"),
            "sample_count": sample_count,
            "total_characters": total_chars,
            "average_length": avg_len,
            "max_length": max_len,
            "min_length": int(min_len) if min_len != float("inf") else 0,
        }

        print(f"\n--- Dataset Inspection: {dataset_name} ---")
        for k, v in results.items():
            print(f"  {k}: {v}")

        return results
