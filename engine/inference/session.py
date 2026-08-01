import uuid
from typing import Any, Dict, List, Optional


class GenerationSession:
    """Maintains stateful cache registries and status for single persistent chat sessions."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.cache: Optional[Any] = None
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False
        if self.cache is not None:
            self.cache.reset()


class ChatSession(GenerationSession):
    """Chat session supporting multi-turn role history tracking and token counting."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        super().__init__(session_id)
        self.messages: List[Dict[str, str]] = []
        self.token_count = 0

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def reset(self) -> None:
        super().reset()
        self.messages.clear()
        self.token_count = 0
