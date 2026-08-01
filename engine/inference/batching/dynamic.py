from typing import List

from engine.inference.request import GenerationRequest


class DynamicBatcher:
    """Manages request merging, padding, and dynamic dispatching to execution models."""

    def __init__(self, max_batch_size: int = 16) -> None:
        self.max_batch_size = max_batch_size

    def merge_requests(
        self, requests: List[GenerationRequest]
    ) -> List[List[GenerationRequest]]:
        """Splits incoming requests into sub-batches based on the max batch limit."""
        batches = []
        for i in range(0, len(requests), self.max_batch_size):
            batches.append(requests[i : i + self.max_batch_size])
        return batches
