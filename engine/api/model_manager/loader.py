from typing import Any, Dict

from engine.inference.service import InferenceService


class ModelLoader:
    """Orchestrates model instantiation and weight checkpoint retrieval."""

    def load(self, model_id: str, config_dict: Dict[str, Any]) -> InferenceService:
        service = InferenceService()
        # Find checkpoint directories or launch default
        checkpoint_dir = config_dict.get("models", {}).get("checkpoint_dir", None)
        service.load_model(checkpoint_dir)
        return service
