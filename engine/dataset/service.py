# ruff: noqa: E501
from typing import Any, Dict, Optional

from engine.dataset.benchmark import DatasetBenchmark
from engine.dataset.cleaner.control_chars import ControlCharacterCleaner
from engine.dataset.cleaner.html import HTMLCleaner
from engine.dataset.cleaner.unicode import UnicodeCleaner
from engine.dataset.cleaner.whitespace import WhitespaceCleaner
from engine.dataset.config import DatasetPipelineConfig
from engine.dataset.inspector import DatasetInspector
from engine.dataset.loader.streaming import StreamingPipeline
from engine.dataset.profiler import DatasetProfiler
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.registry.registry import DatasetRegistry
from engine.dataset.validation import DatasetValidator
from engine.dataset.validator.corruption import CorruptionValidator
from engine.dataset.validator.encoding import EncodingValidator
from engine.dataset.validator.quality import QualityValidator


class DatasetService:
    """Top-level orchestrator for the Chatii dataset engine."""

    def __init__(self, config: Optional[DatasetPipelineConfig] = None) -> None:
        self.config = config or DatasetPipelineConfig.load_from_yaml()
        self.registry = DatasetRegistry()

    def register_dataset(
        self,
        name: str,
        path: str,
        fmt: str = "txt",
        language: str = "en",
        version: str = "v1",
    ) -> None:
        self.registry.register(
            name,
            {
                "name": name,
                "version": version,
                "path": path,
                "format": fmt,
                "language": language,
            },
        )
        print(f"Dataset '{name}' registered successfully.")

    def remove_dataset(self, name: str) -> None:
        if self.registry.remove(name):
            print(f"Dataset '{name}' removed from registry.")
        else:
            print(f"Dataset '{name}' not found in registry.")

    def list_datasets(self) -> None:
        datasets = self.registry.list_all()
        if not datasets:
            print("No datasets registered.")
            return
        print("\n--- Registered Datasets ---")
        for ds in datasets:
            print(
                f"  {ds.get('name', 'unknown')} (v{ds.get('version', '?')}) [{ds.get('format', '?')}] -> {ds.get('path', '?')}"
            )

    def dataset_info(self, name: str) -> None:
        entry = self.registry.get(name)
        if not entry:
            print(f"Dataset '{name}' not found.")
            return
        print(f"\n--- Dataset Info: {name} ---")
        for k, v in entry.items():
            print(f"  {k}: {v}")

    def inspect(self, name: str) -> Dict[str, Any]:
        return DatasetInspector.inspect(name)

    def validate(self, name: str) -> Dict[str, Any]:
        return DatasetValidator.validate(name)

    def benchmark(self, name: str) -> Dict[str, Any]:
        return DatasetBenchmark.benchmark(name)

    def profile(self, name: str) -> Dict[str, Any]:
        return DatasetProfiler.profile(name)

    def prepare(self, name: str) -> None:
        entry = self.registry.get(name)
        if not entry:
            print(f"Dataset '{name}' not found in registry.")
            return

        config = DatasetPipelineConfig(
            name=entry.get("name", name),
            path=entry.get("path", ""),
            loader_type=entry.get("format", "txt"),
        )
        loader = DatasetFactory.create_loader(config)
        cleaners = [
            UnicodeCleaner(),
            HTMLCleaner(),
            ControlCharacterCleaner(),
            WhitespaceCleaner(),
        ]
        validators = [EncodingValidator(), CorruptionValidator(), QualityValidator()]
        pipeline = StreamingPipeline(
            loader=loader, cleaners=cleaners, validators=validators
        )

        count = 0
        for _ in pipeline:
            count += 1

        print(
            f"\nDataset '{name}' prepared successfully. {count} samples passed all filters."
        )

    def stream(self, name: str) -> None:
        entry = self.registry.get(name)
        if not entry:
            print(f"Dataset '{name}' not found in registry.")
            return

        config = DatasetPipelineConfig(
            name=entry.get("name", name),
            path=entry.get("path", ""),
            loader_type=entry.get("format", "txt"),
        )
        loader = DatasetFactory.create_loader(config)
        cleaners = [UnicodeCleaner(), WhitespaceCleaner()]
        pipeline = StreamingPipeline(loader=loader, cleaners=cleaners)

        count = 0
        for sample in pipeline:
            count += 1
            if count <= 5:
                preview = (
                    sample.text[:80] + "..." if len(sample.text) > 80 else sample.text
                )
                print(f"  [{sample.sample_id}] {preview}")

        print(f"\n  ... Streamed {count} samples total.")
