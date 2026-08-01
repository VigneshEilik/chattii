import time
from typing import List


class GenerationRequestState:
    """Maintains token histories and elapsed latency measurements for serving requests."""

    def __init__(self, prompt_token_ids: List[int]) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.generated_token_ids: List[int] = []
        self.start_time = time.perf_counter()
        self.ttft = 0.0

    def record_first_token(self) -> None:
        self.ttft = time.perf_counter() - self.start_time

    def add_token(self, token_id: int) -> None:
        self.generated_token_ids.append(token_id)

    @property
    def total_sequence(self) -> List[int]:
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time
