from typing import Any, List


class TokenizerService:
    """Provides direct token serialization encode/decode utilities."""

    def __init__(self, model_manager: Any) -> None:
        self.model_manager = model_manager

    def encode(self, model_id: str, text: str) -> List[int]:
        service = self.model_manager.get_model(model_id)
        if not service or not service.tokenizer:
            raise KeyError(f"Model '{model_id}' is not loaded or lacks a tokenizer.")
        return service.tokenizer.encode(text)

    def decode(self, model_id: str, token_ids: List[int]) -> str:
        service = self.model_manager.get_model(model_id)
        if not service or not service.tokenizer:
            raise KeyError(f"Model '{model_id}' is not loaded or lacks a tokenizer.")
        return service.tokenizer.decode(token_ids)
