from typing import Any, Dict, Iterator, List, Optional

from engine.inference.config import GenerationConfig
from engine.inference.response import GenerationResponse


class ChatService:
    """Orchestrates formatted message streams and OpenAI chat completions."""

    def __init__(self, model_manager: Any, session_manager: Any) -> None:
        self.model_manager = model_manager
        self.session_manager = session_manager

    def generate_chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        session_id: Optional[str] = None,
    ) -> GenerationResponse:
        service = self.model_manager.get_model(model_id)
        if not service:
            raise KeyError(f"Model '{model_id}' is not loaded.")

        # Format chat history
        session = service.new_session(session_id=session_id, is_chat=True)
        session.messages = messages

        # Execute generate
        return service.chat(session, config=config)

    def stream_chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        service = self.model_manager.get_model(model_id)
        if not service:
            raise KeyError(f"Model '{model_id}' is not loaded.")

        # Tokenize and format using ChatFormatter
        prompt_tokens = service.formatter.format_and_tokenize(messages, max_length=2048)
        prompt_str = service.tokenizer.decode(prompt_tokens)

        # Execute streaming generator
        yield from service.stream(prompt_str, config=config)
