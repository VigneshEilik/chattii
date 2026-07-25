from pathlib import Path
from typing import Union

import yaml

from engine.training.config.schema import (
    CheckpointConfig,
    LoggingConfig,
    ModelTrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    ValidationConfig,
)


class TrainingConfigLoader:
    """Loads and compiles configurations from configs/*.yaml files."""

    @staticmethod
    def load_from_yaml(config_dir: Union[str, Path] = "configs") -> ModelTrainingConfig:
        """
        Loads training, optimizer, and scheduler settings from config files
        and compiles them into a unified ModelTrainingConfig.
        """
        dir_path = Path(config_dir)

        # 1. Load training.yaml settings
        training_path = dir_path / "training.yaml"
        training_data = {}
        if training_path.exists():
            with open(training_path, "r", encoding="utf-8") as f:
                training_data = yaml.safe_load(f) or {}

        # 2. Load optimizer.yaml settings
        optimizer_path = dir_path / "optimizer.yaml"
        optimizer_data = {}
        if optimizer_path.exists():
            with open(optimizer_path, "r", encoding="utf-8") as f:
                optimizer_data = yaml.safe_load(f) or {}

        # 3. Load scheduler.yaml settings
        scheduler_path = dir_path / "scheduler.yaml"
        scheduler_data = {}
        if scheduler_path.exists():
            with open(scheduler_path, "r", encoding="utf-8") as f:
                scheduler_data = yaml.safe_load(f) or {}

        # Map training.yaml keys
        compiled_data = {
            "seed": training_data.get("seed", 42),
            "batch_size": training_data.get("batch_size", 8),
            "gradient_accumulation_steps": training_data.get(
                "gradient_accumulation_steps", 1
            ),
            "epochs": training_data.get("max_epochs", 5),
            "max_steps": training_data.get("max_steps"),
            "mixed_precision": training_data.get("mixed_precision", "none"),
            "grad_clip": training_data.get("grad_clip", 1.0),
            "device": training_data.get("device", "auto"),
        }

        # Map optimizer settings
        opt_betas = optimizer_data.get("betas", [0.9, 0.95])
        compiled_data["optimizer"] = OptimizerConfig(
            name=optimizer_data.get("type", "adamw"),
            lr=optimizer_data.get("learning_rate", 3e-4),
            weight_decay=optimizer_data.get("weight_decay", 0.1),
            beta1=opt_betas[0] if len(opt_betas) > 0 else 0.9,
            beta2=opt_betas[1] if len(opt_betas) > 1 else 0.95,
            eps=optimizer_data.get("eps", 1e-8),
        )

        # Map scheduler settings
        compiled_data["scheduler"] = SchedulerConfig(
            name=scheduler_data.get("type", "cosine"),
            warmup_steps=scheduler_data.get("warmup_steps", 1000),
            min_lr=scheduler_data.get("min_lr", 3e-5),
        )

        # Map checkpoint settings
        compiled_data["checkpoint"] = CheckpointConfig(
            save_every_steps=training_data.get("save_interval", 1000),
            dir=training_data.get("checkpoint_dir", "models/checkpoints"),
            keep_top_n=training_data.get("keep_top_n", 3),
        )

        # Map validation settings
        compiled_data["validation"] = ValidationConfig(
            every_steps=training_data.get("eval_interval", 1000),
            early_stopping_patience=training_data.get("early_stopping_patience", 3),
        )

        # Map logging settings
        compiled_data["logging"] = LoggingConfig(
            tensorboard=training_data.get("tensorboard", True),
            csv=training_data.get("csv", True),
            console=training_data.get("console", True),
            dashboard=training_data.get("dashboard", True),
        )

        return ModelTrainingConfig(**compiled_data)
