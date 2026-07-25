import os

from core.utils.config import ChatiiConfig


def test_load_config_defaults_and_yaml(temp_config_dir: str) -> None:
    """
    Verifies that configurations load correctly from YAML,
    merging with defaults for fields not declared in the YAML.
    """
    config = ChatiiConfig.load_from_dir(temp_config_dir)

    # Overwritten by YAML
    assert config.model.name == "test-chatii"
    assert config.model.hidden_dim == 128
    assert config.model.num_layers == 4
    assert config.training.batch_size == 16

    # Falls back to default values in Pydantic schema
    assert config.model.num_heads == 8
    assert config.optimizer.type == "adamw"


def test_env_variable_overrides(temp_config_dir: str) -> None:
    """
    Verifies that env variables prefixed with CHATII_ override
    yaml properties (supports both single and double underscores).
    """
    os.environ["CHATII_MODEL_HIDDEN_DIM"] = "512"
    os.environ["CHATII_TRAINING__BATCH_SIZE"] = "64"

    try:
        config = ChatiiConfig.load_from_dir(temp_config_dir)
        assert config.model.hidden_dim == 512
        assert config.training.batch_size == 64
    finally:
        # Clean up env variables to avoid polluting other tests
        os.environ.pop("CHATII_MODEL_HIDDEN_DIM", None)
        os.environ.pop("CHATII_TRAINING__BATCH_SIZE", None)


def test_runtime_updates(temp_config_dir: str) -> None:
    """
    Verifies that runtime config adjustments are applied and validated.
    """
    config = ChatiiConfig.load_from_dir(temp_config_dir)
    assert config.training.batch_size == 16

    # Update runtime parameters
    config.update_runtime({"training": {"batch_size": 128}})
    assert config.training.batch_size == 128
