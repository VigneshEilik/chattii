import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

# --- Sub-Configuration Models ---


class ModelConfig(BaseModel):
    name: str = "chatii-10m"
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 684
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.1


class TrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    max_steps: Optional[int] = None
    mixed_precision: str = "none"
    grad_clip: float = 1.0
    eval_interval: int = 500
    log_interval: int = 50
    save_interval: int = 1000
    device: str = "cuda"


class OptimizerConfig(BaseModel):
    type: str = "adamw"
    learning_rate: float = 5.0e-4
    betas: List[float] = [0.9, 0.95]
    eps: float = 1.0e-8
    weight_decay: float = 0.1


class SchedulerConfig(BaseModel):
    type: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 5.0e-5
    decay_steps: Optional[int] = None


class TokenizerConfig(BaseModel):
    type: str = "bpe"
    vocab_size: int = 32000
    vocab_file: str = "models/registry/tokenizer.model"
    special_tokens: Dict[str, str] = {
        "pad": "<pad>",
        "unk": "<unk>",
        "bos": "<s>",
        "eos": "</s>",
    }
    lowercase: bool = False


class DatasetConfig(BaseModel):
    name: str = "tinystories"
    path: str = "datasets/download/tinystories"
    format: str = "text"
    splits: Dict[str, str] = {"train": "train.txt", "validation": "validation.txt"}
    tokenized_path: str = "datasets/tokenized/tinystories"
    cache_dir: str = "datasets/cache"
    validation_split_ratio: float = 0.05
    seed: int = 42


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    api_key_header: str = "X-API-Key"
    rate_limit: Dict[str, Any] = {"enabled": True, "requests_per_minute": 60}
    max_batch_size: int = 8
    timeout_seconds: float = 5.0
    enable_streaming: bool = True


class LoggingConfig(BaseModel):
    version: int = 1
    disable_existing_loggers: bool = False
    level: str = "INFO"
    format: str = "JSON"
    file_path: Optional[str] = "chatii.log"
    enable_console: bool = True


# --- Main Configuration Class ---


class ChatiiConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_from_dir(cls, config_dir: str = "configs") -> "ChatiiConfig":
        """
        Loads yaml configuration files from a directory, merges them,
        applies env variable overrides, and returns the validated config object.
        """
        config_path = Path(config_dir)
        yaml_data = {}

        # 1. Load environment variables from .env files
        load_all_envs()

        # 2. Parse YAML files
        for section in cls.model_fields.keys():
            file_path = config_path / f"{section}.yaml"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        yaml_content = yaml.safe_load(f) or {}
                        yaml_data[section] = yaml_content
                    except yaml.YAMLError as e:
                        raise ValueError(
                            f"Failed to parse configuration file {file_path}: {e}"
                        ) from e

        # 3. Create the initial validated pydantic object
        config = cls.model_validate(yaml_data)

        # 4. Apply environment variable overrides (e.g., CHATII_MODEL__HIDDEN_DIM=512)
        env_overrides = get_env_overrides()
        for section, fields in env_overrides.items():
            if hasattr(config, section):
                sub_config = getattr(config, section)
                for field, val in fields.items():
                    if hasattr(sub_config, field):
                        setattr(sub_config, field, val)

                # Re-validate the overridden sub-config
                validated_sub = sub_config.__class__.model_validate(
                    sub_config.model_dump()
                )
                setattr(config, section, validated_sub)

        return config

    def update_runtime(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Dynamically update config values at runtime with Pydantic validation.
        """
        for section, fields in updates.items():
            if hasattr(self, section):
                sub_config = getattr(self, section)
                for field, val in fields.items():
                    if hasattr(sub_config, field):
                        setattr(sub_config, field, val)

                # Re-validate the updated sub-config
                validated_sub = sub_config.__class__.model_validate(
                    sub_config.model_dump()
                )
                setattr(self, section, validated_sub)


# --- Env Loading Helper Functions ---


def load_dotenv(dotenv_path: Path, overwrite: bool = True) -> None:
    """Reads a .env file and updates os.environ."""
    if not dotenv_path.exists():
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Remove surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if overwrite or key not in os.environ:
                    os.environ[key] = val


def load_all_envs() -> None:
    """Loads env files with appropriate override ordering."""
    # 1. Load env-specific first (e.g., .env.development)
    env = os.environ.get("CHATII_ENV", "development")
    load_dotenv(Path(f".env.{env}"), overwrite=True)

    # 2. Load local .env second (overrides specific profiles)
    load_dotenv(Path(".env"), overwrite=True)


def parse_env_value(value: str) -> Any:
    """Helper to convert environment strings to appropriate Python primitives."""
    val_lower = value.lower()
    if val_lower in ("true", "yes", "on"):
        return True
    if val_lower in ("false", "no", "off"):
        return False
    if val_lower in ("null", "none"):
        return None

    # Try parsing numeric types
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass

    return value


def get_env_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Scans environment variables starting with CHATII_
    and groups them into {section: {field: value}}.
    Supports both CHATII_MODEL__HIDDEN_DIM and CHATII_MODEL_HIDDEN_DIM format.
    """
    overrides: Dict[str, Dict[str, Any]] = {}
    sections = {
        "model",
        "training",
        "optimizer",
        "scheduler",
        "tokenizer",
        "dataset",
        "api",
        "logging",
    }

    for key, value in os.environ.items():
        if key.startswith("CHATII_"):
            # Strip CHATII_ prefix
            raw_key = key[7:].lower()

            section = None
            field = None

            # Check double underscore format first (CHATII_MODEL__HIDDEN_DIM)
            if "__" in raw_key:
                parts = raw_key.split("__", 1)
                if parts[0] in sections:
                    section, field = parts[0], parts[1]
            else:
                # Fallback to single underscore prefix checking
                for sec in sections:
                    if raw_key.startswith(f"{sec}_"):
                        section = sec
                        field = raw_key[len(sec) + 1 :]
                        break

            if section and field:
                if section not in overrides:
                    overrides[section] = {}
                overrides[section][field] = parse_env_value(value)

    return overrides
