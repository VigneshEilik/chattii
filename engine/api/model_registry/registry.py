from typing import Dict, List, Optional

from engine.api.model_registry.metadata import ModelMetadata


class ModelRegistry:
    """Registry class managing static checkpoint configurations and versions."""

    def __init__(self) -> None:
        self.available_models: Dict[str, ModelMetadata] = {
            "chatii-v1": ModelMetadata(
                model_id="chatii-v1",
                version="1.0.0",
                description="Chatii production model v1",
                max_seq_len=2048,
                d_model=768,
                num_layers=12,
            )
        }

    def list_models(self) -> List[ModelMetadata]:
        return list(self.available_models.values())

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        return self.available_models.get(model_id)
