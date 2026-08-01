from typing import Any


class BeforeDecodeHook:
    """Invoked prior to decoding next sequence token."""

    def __call__(self, step: int, logits: Any) -> None:
        pass
