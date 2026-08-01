from typing import Optional

from sdk.python.chatii_sdk.chat import Chat
from sdk.python.chatii_sdk.completion import Completion
from sdk.python.chatii_sdk.embedding import Embedding
from sdk.python.chatii_sdk.models import Models


class ChatiiClient:
    """Production Python SDK client wrapper accessing Chatii APIs."""

    def __init__(
        self, base_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key

        self.chat = Chat(self)
        self.completions = Completion(self)
        self.embeddings = Embedding(self)
        self.models = Models(self)
