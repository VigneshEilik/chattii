from typing import List

from engine.inference.tracing.span import Span


class RequestTrace:
    """Orchestrates nesting trace spans for a single generation request lifecycle."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.spans: List[Span] = []

    def start_span(self, name: str) -> Span:
        span = Span(name)
        self.spans.append(span)
        return span
