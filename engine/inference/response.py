from typing import List


class GenerationResponse:
    """Encapsulates execution text and performance tracking metrics."""

    def __init__(
        self,
        text: str,
        tokens: List[str],
        token_ids: List[int],
        finish_reason: str,  # eos_token, length, stop_sequence, cancelled, timeout, error
        prompt_tokens: int,
        generated_tokens: int,
        latency_seconds: float,
        tokens_per_second: float,
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.latency_seconds = latency_seconds
        self.tokens_per_second = tokens_per_second
