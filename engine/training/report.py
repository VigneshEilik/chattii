from pathlib import Path
from typing import Any, Dict, Union


class TrainingReportGenerator:
    """Generates markdown training reports and model cards for HF registry."""

    @staticmethod
    def generate_report(
        save_path: Union[str, Path],
        run_id: str,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> None:
        """Writes details on hyperparams, speeds, and losses to training_report.md."""
        report = f"""# Training Report: {run_id}

## Summary
- **Final Loss**: {metrics.get("loss", 0.0):.4f}
- **Best Loss**: {metrics.get("best_loss", 0.0):.4f}
- **Best Perplexity**: {metrics.get("best_perplexity", 0.0):.4f}
- **Total Tokens Seen**: {metrics.get("tokens_seen", 0):,}
- **Total Epochs**: {config.get("epochs", 0)}

## Configuration
- **Batch Size**: {config.get("batch_size", 8)}
- **Sequence Length**: {config.get("sequence_length", 1024)}
- **Learning Rate**: {config.get("optimizer", {}).get("lr", 3e-4):.2e}
- **Optimizer**: {config.get("optimizer", {}).get("name", "adamw")}
"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

    @staticmethod
    def generate_model_card(
        save_path: Union[str, Path], model_name: str, metrics: Dict[str, Any]
    ) -> None:
        """Writes model validation outcomes and usability scopes to model_card.md."""
        card = f"""# Model Card: {model_name}

## Description
This model was trained using the Chatii production training engine.

## Training Metrics
- **Validation Loss**: {metrics.get("val_loss", 0.0):.4f}
- **Validation Perplexity**: {metrics.get("val_perplexity", 0.0):.4f}
- **Validation Accuracy**: {metrics.get("val_accuracy", 0.0):.2%}
"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(card)

        # Save a flat JSON report as well
        report_json_path = Path(save_path).parent / "report.json"
        import json

        with open(report_json_path, "w", encoding="utf-8") as f_json:
            json.dump(metrics, f_json, indent=2)
