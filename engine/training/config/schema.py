from typing import Optional

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    """Configuration schema for custom optimizers (AdamW, SGD)."""

    name: str = Field(default="adamw")
    lr: float = Field(default=3e-4)
    weight_decay: float = Field(default=0.1)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.95)
    eps: float = Field(default=1e-8)


class SchedulerConfig(BaseModel):
    """Configuration schema for learning rate schedulers."""

    name: str = Field(default="cosine")
    warmup_steps: int = Field(default=1000)
    min_lr: float = Field(default=3e-5)


class CheckpointConfig(BaseModel):
    """Configuration schema for checkpoint serialization options."""

    save_every_steps: int = Field(default=1000)
    dir: str = Field(default="models/checkpoints")
    keep_top_n: int = Field(default=3)


class ValidationConfig(BaseModel):
    """Configuration schema for training validation loops."""

    every_steps: int = Field(default=1000)
    early_stopping_patience: int = Field(default=3)


class LoggingConfig(BaseModel):
    """Configuration schema for training metric logging outputs."""

    tensorboard: bool = Field(default=True)
    csv: bool = Field(default=True)
    console: bool = Field(default=True)
    dashboard: bool = Field(default=True)


class PreprocessingConfig(BaseModel):
    """Configuration schema for automatic raw dataset cleaning."""

    clean: bool = Field(default=True)
    deduplicate: bool = Field(default=True)
    language_filter: Optional[str] = Field(default="en")
    quality_filter: bool = Field(default=True)


class ModelTrainingConfig(BaseModel):
    """Main training configuration schema orchestrating all components."""

    seed: int = Field(default=42)
    batch_size: int = Field(default=8)
    sequence_length: int = Field(default=1024)
    gradient_accumulation_steps: int = Field(default=1)
    epochs: int = Field(default=5)
    max_steps: Optional[int] = Field(default=None)
    mixed_precision: str = Field(default="none")  # options: none, fp16, bf16
    grad_clip: float = Field(default=1.0)
    device: str = Field(default="auto")

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
