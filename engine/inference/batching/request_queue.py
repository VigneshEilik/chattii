from typing import List

from engine.inference.request import GenerationRequest, RequestState


class RequestQueue:
    """Manates scheduling, cancellations, and prioritization of incoming client requests."""

    def __init__(self) -> None:
        self.queue: List[GenerationRequest] = []

    def enqueue(self, request: GenerationRequest) -> None:
        request.state = RequestState.PENDING
        self.queue.append(request)

    def get_next_batch(self, batch_size: int) -> List[GenerationRequest]:
        # Sort queue by priority descending, then FCFS
        self.queue.sort(key=lambda r: r.priority, reverse=True)
        batch = []
        remaining = []
        for r in self.queue:
            if r.cancelled:
                r.state = RequestState.CANCELLED
                continue
            if len(batch) < batch_size:
                r.state = RequestState.PREFILL
                batch.append(r)
            else:
                remaining.append(r)
        self.queue = remaining
        return batch

    def cancel(self, session_id: str) -> None:
        for r in self.queue:
            if r.session_id == session_id:
                r.cancelled = True
                r.state = RequestState.CANCELLED

    def size(self) -> int:
        return len([r for r in self.queue if not r.cancelled])
