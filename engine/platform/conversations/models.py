import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Attachment:
    id: str
    filename: str
    file_type: str
    size_bytes: int
    url: Optional[str] = None
    extracted_text: Optional[str] = None


@dataclass
class ToolCall:
    tool_id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None


@dataclass
class Citation:
    document_name: str
    chunk_index: int
    snippet: str
    score: float


@dataclass
class Message:
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: float = field(default_factory=time.time)
    model: Optional[str] = None
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    latency_seconds: float = 0.0
    finish_reason: Optional[str] = "stop"
    attachments: List[Attachment] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    user_id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = "chatii-v1"
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    message_count: int = 0
