import time
from typing import List


class StoppingCriteria:
    """Base interface for checking generation halting rules."""

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        raise NotImplementedError


class StopOnEOS(StoppingCriteria):
    """Halts generation if the last token matches an EOS token ID."""

    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        return last_token == self.eos_token_id


class StopOnTokens(StoppingCriteria):
    """Halts generation if cumulative sequence length exceeds maximum token count."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        return len(input_ids) >= self.max_tokens


class StopOnTime(StoppingCriteria):
    """Halts generation if runtime exceeds threshold."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        return (time.time() - self.start_time) >= self.timeout_seconds


class StopOnString(StoppingCriteria):
    """Halts generation if a specific string sequence is detected in output (stub)."""

    def __init__(self, stop_strings: List[str]) -> None:
        self.stop_strings = stop_strings

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        return False


class CompositeStoppingCriteria(StoppingCriteria):
    """Aggregates multiple stopping rules; returns True if any rule triggers."""

    def __init__(self, criteria_list: List[StoppingCriteria]) -> None:
        self.criteria_list = criteria_list

    def __call__(self, input_ids: List[int], last_token: int) -> bool:
        return any(criteria(input_ids, last_token) for criteria in self.criteria_list)
