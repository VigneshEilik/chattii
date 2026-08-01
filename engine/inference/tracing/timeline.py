from typing import List

from engine.inference.tracing.span import Span


class TraceTimeline:
    """ASCII-based timeline generator for request spans."""

    def format_timeline(self, spans: List[Span]) -> str:
        # Simple ASCII representation of spans
        result = []
        for s in spans:
            result.append(f"{s.name}: {s.duration * 1000:.2f}ms")
        return "\n".join(result)
