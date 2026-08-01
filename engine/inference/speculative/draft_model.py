from typing import Any


class DraftModelWrapper:
    """Stub wrapper interface for fast speculative decoding draft models."""

    def __init__(self, draft_model: Any) -> None:
        self.draft_model = draft_model
