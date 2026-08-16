import uuid
from typing import Dict, List, Optional

from engine.platform.conversations.models import Conversation, Message, MessageRole


class ConversationRepository:
    """Thread-safe storage for conversations and message histories."""

    def __init__(self) -> None:
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[Message]] = {}

    def create_conversation(
        self,
        user_id: str,
        title: str = "New Conversation",
        model: str = "chatii-v1",
        system_prompt: Optional[str] = None,
    ) -> Conversation:
        conv_id = f"conv-{uuid.uuid4().hex[:12]}"
        conv = Conversation(
            id=conv_id,
            user_id=user_id,
            title=title,
            model=model,
            system_prompt=system_prompt,
        )
        self._conversations[conv_id] = conv
        self._messages[conv_id] = []
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self._conversations.get(conv_id)

    def list_user_conversations(self, user_id: str) -> List[Conversation]:
        return [
            c
            for c in sorted(
                self._conversations.values(),
                key=lambda x: x.updated_at,
                reverse=True,
            )
            if c.user_id == user_id
        ]

    def rename_conversation(
        self, conv_id: str, new_title: str
    ) -> Optional[Conversation]:
        conv = self._conversations.get(conv_id)
        if conv:
            conv.title = new_title.strip()
            return conv
        return None

    def delete_conversation(self, conv_id: str) -> bool:
        conv = self._conversations.pop(conv_id, None)
        if conv:
            self._messages.pop(conv_id, None)
            return True
        return False

    def add_message(
        self,
        conv_id: str,
        role: MessageRole,
        content: str,
        model: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
        latency_seconds: float = 0.0,
    ) -> Message:
        conv = self._conversations.get(conv_id)
        if not conv:
            raise ValueError(f"Conversation '{conv_id}' not found.")

        msg_id = f"msg-{uuid.uuid4().hex[:12]}"
        msg = Message(
            id=msg_id,
            conversation_id=conv_id,
            role=role,
            content=content,
            model=model or conv.model,
            token_usage=token_usage or {"prompt_tokens": 0, "completion_tokens": 0},
            latency_seconds=latency_seconds,
        )
        self._messages[conv_id].append(msg)
        conv.message_count += 1
        conv.updated_at = msg.created_at
        return msg

    def get_messages(self, conv_id: str) -> List[Message]:
        return self._messages.get(conv_id, [])


_conv_repo_instance: Optional[ConversationRepository] = None


def get_conversation_repository() -> ConversationRepository:
    global _conv_repo_instance
    if _conv_repo_instance is None:
        _conv_repo_instance = ConversationRepository()
    return _conv_repo_instance
