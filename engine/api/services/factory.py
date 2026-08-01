from engine.api.services.chat_service import ChatService
from engine.api.services.completion_service import CompletionService
from engine.api.services.embedding_service import EmbeddingService
from engine.api.services.tokenizer_service import TokenizerService
from engine.api.state.state import ApplicationState


class ApiServiceFactory:
    """Dependency injection factory providing instantiated chat, completions, and tokenizer services."""

    @staticmethod
    def get_chat_service(state: ApplicationState) -> ChatService:
        return ChatService(state.model_manager, state.session_manager)

    @staticmethod
    def get_completion_service(state: ApplicationState) -> CompletionService:
        return CompletionService(state.model_manager, state.session_manager)

    @staticmethod
    def get_embedding_service(state: ApplicationState) -> EmbeddingService:
        return EmbeddingService(state.model_manager)

    @staticmethod
    def get_tokenizer_service(state: ApplicationState) -> TokenizerService:
        return TokenizerService(state.model_manager)
