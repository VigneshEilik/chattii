from typing import List, Set


class RegexConstraint:
    """Filters logits to match a target regular expression constraint pattern."""

    def __init__(self, regex_pattern: str) -> None:
        self.regex_pattern = regex_pattern

    def get_allowed_tokens(self, current_ids: List[int], vocab_size: int) -> Set[int]:
        # Stub interface allowing all tokens
        return set(range(vocab_size))
