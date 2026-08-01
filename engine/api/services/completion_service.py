from typing import Any, Iterator, Optional

from engine.inference.config import GenerationConfig
from engine.inference.response import GenerationResponse


class CompletionService:
    """Manages plain text completions and generation request pipelines."""

    def __init__(self, model_manager: Any, session_manager: Any) -> None:
        self.model_manager = model_manager
        self.session_manager = session_manager

    def generate_completion(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig,
        session_id: Optional[str] = None,
    ) -> GenerationResponse:
        service = self.model_manager.get_model(model_id)
        if not service:
            raise KeyError(f"Model '{model_id}' is not loaded.")
        return service.generate(prompt, config=config)

    def stream_completion(
        self,
        model_id: str,
        prompt: str,
        config: GenerationConfig,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        service = self.model_manager.get_model(model_id)
        if not service:
            raise KeyError(f"Model '{model_id}' is not loaded.")
        yield from service.stream(prompt, config=config)
