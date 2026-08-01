from typing import Dict, Optional

from engine.api.model_manager.loader import ModelLoader
from engine.api.model_manager.unloader import ModelUnloader
from engine.api.model_registry.registry import ModelRegistry
from engine.inference.service import InferenceService


class ModelManager:
    """Manages active loaded model instances, aliases, versioning, and hotswaps."""

    def __init__(self, config_dict: dict) -> None:
        self.config_dict = config_dict
        self.registry = ModelRegistry()
        self.loader = ModelLoader()
        self.unloader = ModelUnloader()
        self.loaded_models: Dict[str, InferenceService] = {}

        # Load default model
        default_name = config_dict.get("models", {}).get("default_model", "chatii-v1")
        self.load_model(default_name)

    def load_model(self, model_id: str) -> InferenceService:
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        metadata = self.registry.get_metadata(model_id)
        if not metadata:
            # Fallback for dynamic/test model registrations
            metadata = self.registry.available_models["chatii-v1"]

        service = self.loader.load(model_id, self.config_dict)
        self.loaded_models[model_id] = service
        return service

    def unload_model(self, model_id: str) -> None:
        if model_id in self.loaded_models:
            service = self.loaded_models.pop(model_id)
            self.unloader.unload(service)

    def reload_model(self, model_id: str) -> InferenceService:
        self.unload_model(model_id)
        return self.load_model(model_id)

    def unload_all(self) -> None:
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)

    def get_model(self, model_id: str) -> Optional[InferenceService]:
        # Handle aliases
        aliases = self.config_dict.get("models", {}).get("model_aliases", {})
        target_id = aliases.get(model_id, model_id)

        if target_id not in self.loaded_models:
            return self.load_model(target_id)
        return self.loaded_models[target_id]

    def warmup_models(self) -> None:
        for service in self.loaded_models.values():
            service.warmup()
