import os
import tempfile
from engine.dataset.registry.registry import DatasetRegistry
from engine.dataset.registry.factory import DatasetFactory
from engine.dataset.config import DatasetPipelineConfig


def test_dataset_registry() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_file = os.path.join(temp_dir, "registry.json")
        registry = DatasetRegistry(registry_file=registry_file)

        # Confirm registry seeds default
        assert registry.get("tinystories") is not None

        # Register custom dataset
        config = {
            "name": "custom_data",
            "version": "v1",
            "path": "path/to/data.txt",
            "format": "txt",
        }
        registry.register("custom_data", config)

        assert registry.get("custom_data") == config

        # Remove dataset
        assert registry.remove("custom_data")
        assert registry.get("custom_data") is None


def test_dataset_factory() -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("Line")
        temp_name = f.name

    try:
        config = DatasetPipelineConfig(path=temp_name, loader_type="txt")
        loader = DatasetFactory.create_loader(config)
        assert loader is not None
        assert loader.path == temp_name
    finally:
        os.remove(temp_name)
