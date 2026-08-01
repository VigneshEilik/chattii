from enum import Enum
from typing import List, Optional

from engine.inference.config import GenerationConfig


class RequestState(Enum):
    """Lifecycle states for active serving requests."""

    PENDING = "PENDING"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    WAITING = "WAITING"
    STREAMING = "STREAMING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class GenerationRequest:
    """Encapsulates client prompt context and generation params for serving queues."""

    def __init__(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        stop_words: Optional[List[str]] = None,
        priority: int = 0,
    ) -> None:
        self.prompt = prompt
        self.config = config or GenerationConfig()
        self.session_id = session_id
        self.seed = seed
        self.stop_words = stop_words or []
        self.priority = priority
        self.state = RequestState.PENDING
        self.cancelled = False
