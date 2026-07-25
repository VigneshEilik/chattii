# ruff: noqa: E501, C901
import torch

from engine.training.config.schema import ModelTrainingConfig
from engine.training.exceptions import ConfigurationError


class TrainingConfigValidator:
    """Validates ModelTrainingConfig settings."""

    @staticmethod
    def validate(config: ModelTrainingConfig) -> None:
        """
        Validates hyperparameter consistency and raises ConfigurationError
        on any violations.
        """
        if config.seed < 0:
            raise ConfigurationError(
                f"Random seed must be non-negative, got {config.seed}"
            )

        if config.batch_size <= 0:
            raise ConfigurationError(
                f"Batch size must be greater than 0, got {config.batch_size}"
            )

        if config.gradient_accumulation_steps <= 0:
            raise ConfigurationError(
                f"Gradient accumulation steps must be >= 1, got {config.gradient_accumulation_steps}"
            )

        if config.epochs <= 0:
            raise ConfigurationError(
                f"Epochs count must be greater than 0, got {config.epochs}"
            )

        if config.mixed_precision not in ["none", "fp16", "bf16"]:
            raise ConfigurationError(
                f"Invalid mixed precision: '{config.mixed_precision}'. Options: none, fp16, bf16"
            )

        if config.device not in ["auto", "cpu", "cuda", "mps"]:
            raise ConfigurationError(
                f"Invalid device: '{config.device}'. Options: auto, cpu, cuda, mps"
            )

        # Validate optimizer settings
        opt = config.optimizer
        if opt.lr <= 0:
            raise ConfigurationError(f"Learning rate must be positive, got {opt.lr}")
        if opt.weight_decay < 0:
            raise ConfigurationError(
                f"Weight decay cannot be negative, got {opt.weight_decay}"
            )
        if not (0 <= opt.beta1 < 1.0) or not (0 <= opt.beta2 < 1.0):
            raise ConfigurationError(
                f"Optimizer betas must be in [0, 1), got betas=({opt.beta1}, {opt.beta2})"
            )

        # Validate scheduler settings
        sched = config.scheduler
        if sched.warmup_steps < 0:
            raise ConfigurationError(
                f"Warmup steps cannot be negative, got {sched.warmup_steps}"
            )
        if sched.min_lr < 0:
            raise ConfigurationError(
                f"Min learning rate cannot be negative, got {sched.min_lr}"
            )

        # Check CUDA availability if requested
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA device requested but CUDA is not available on this host."
            )
