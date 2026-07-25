import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml


@pytest.fixture
def temp_config_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory populated with minimal valid yaml configs
    for configuration loading unit tests.
    """
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir)

    # 1. Write minimal config files
    with open(config_path / "model.yaml", "w") as f:
        yaml.dump({"name": "test-chatii", "hidden_dim": 128, "num_layers": 4}, f)

    with open(config_path / "training.yaml", "w") as f:
        yaml.dump({"seed": 42, "batch_size": 16}, f)

    with open(config_path / "logging.yaml", "w") as f:
        yaml.dump({"level": "DEBUG", "format": "JSON", "enable_console": True}, f)

    # Populate others with empty dictionaries to load default values
    for section in ["optimizer", "scheduler", "tokenizer", "dataset", "api"]:
        with open(config_path / f"{section}.yaml", "w") as f:
            yaml.dump({}, f)

    yield temp_dir

    # Clean up directory after test completes
    shutil.rmtree(temp_dir)
