from typing import List


class TokenBudgetManager:
    """Manages prompt truncation, reserved tokens, and available generation tokens."""

    def __init__(
        self,
        max_context: int = 2048,
        max_new_tokens: int = 256,
        reserved_tokens: int = 4,
    ) -> None:
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.reserved_tokens = reserved_tokens

    def get_available_budget(self, prompt_len: int) -> int:
        remaining_context = self.max_context - prompt_len - self.reserved_tokens
        return max(0, min(self.max_new_tokens, remaining_context))

    def truncate_prompt(self, input_ids: List[int]) -> List[int]:
        limit = self.max_context - self.reserved_tokens - 1
        if len(input_ids) > limit:
            return input_ids[-limit:]
        return input_ids
