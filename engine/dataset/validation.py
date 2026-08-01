# ruff: noqa: E501
import json
import os
from typing import Any, Dict

from engine.dataset.cleaner.control_chars import ControlCharacterCleaner
from engine.dataset.cleaner.html import HTMLCleaner
from engine.dataset.cleaner.pipeline import CleaningPipeline
from engine.dataset.cleaner.unicode import UnicodeCleaner
from engine.dataset.cleaner.whitespace import WhitespaceCleaner
from engine.dataset.config import DatasetPipelineConfig
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.registry.registry import DatasetRegistry
from engine.dataset.validator.corruption import CorruptionValidator
from engine.dataset.validator.duplicates import DuplicateValidator
from engine.dataset.validator.encoding import EncodingValidator
from engine.dataset.validator.quality import QualityValidator


class DatasetValidator:
    """Runs the full cleaning and validation pipeline on a registered dataset and produces validation_report.json."""

    @staticmethod
    def validate(dataset_name: str, output_dir: str = "datasets") -> Dict[str, Any]:
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

        cleaning_pipeline = CleaningPipeline(
            cleaners=[
                UnicodeCleaner(),
                HTMLCleaner(),
                ControlCharacterCleaner(),
                WhitespaceCleaner(),
            ]
        )

        encoding_validator = EncodingValidator()
        corruption_validator = CorruptionValidator()
        quality_validator = QualityValidator(
            min_length=config.min_length, max_length=config.max_length
        )
        duplicate_validator = DuplicateValidator(
            exact=config.deduplicate, near_dup=False
        )

        total = 0
        passed = 0
        failed = 0
        reasons: Dict[str, int] = {}

        for sample in loader:
            total += 1
            cleaned = cleaning_pipeline.clean(sample)

            validators = [
                encoding_validator,
                corruption_validator,
                quality_validator,
                duplicate_validator,
            ]
            is_valid = True
            for v in validators:
                valid, reason = v.validate(cleaned)
                if not valid:
                    is_valid = False
                    reasons[reason] = reasons.get(reason, 0) + 1
                    break

            if is_valid:
                passed += 1
            else:
                failed += 1

        report = {
            "dataset_name": dataset_name,
            "total_samples": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / max(1, total)) * 100,
            "failure_reasons": reasons,
        }

        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n--- Validation Report: {dataset_name} ---")
        print(f"  Total:  {total}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Pass Rate: {report['pass_rate']:.2f}%")
        print(f"  Report saved to: {report_path}")

        return report
