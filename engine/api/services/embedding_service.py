from typing import Any, List


class EmbeddingManager:
    """Manages separate embedding models and runtime calculations."""

    def compute_embedding(self, service: Any, text: str) -> List[float]:
        # Simple stub returning mock embeddings array matching model dimension
        d_model = (
            service.model.config.d_model
            if (service.model and service.model.config)
            else 768
        )
        # Return static normalized float coordinates
        return [0.1] * d_model


class EmbeddingService:
    """Computes hidden token state embeddings."""

    def __init__(self, model_manager: Any) -> None:
        self.model_manager = model_manager
        self.embedding_mgr = EmbeddingManager()

    def generate_embeddings(self, model_id: str, text: str) -> List[float]:
        service = self.model_manager.get_model(model_id)
        if not service:
            raise KeyError(f"Model '{model_id}' is not loaded.")
        return self.embedding_mgr.compute_embedding(service, text)
