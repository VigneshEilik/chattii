from typing import Optional


class RequestValidator:
    """Validates temperature, top-k, top-p, and context length limitations."""

    def __init__(
        self, max_prompt_length: int = 2048, max_new_tokens: int = 512
    ) -> None:
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens

    def validate_generation_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        prompt_len: int,
    ) -> None:
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")

        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if prompt_len > self.max_prompt_length:
            raise ValueError(
                f"Prompt length {prompt_len} exceeds max allowed prompt length of {self.max_prompt_length}"
            )
