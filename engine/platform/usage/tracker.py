import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UsageRecord:
    id: str
    user_id: str
    key_id: Optional[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)


class UsageTracker:
    """Tracks token usage, requests, and latency across users and API keys."""

    def __init__(self) -> None:
        self._records: List[UsageRecord] = []

    def record_usage(
        self,
        user_id: str,
        key_id: Optional[str],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_ms: float,
        total_latency_ms: float,
    ) -> UsageRecord:
        record = UsageRecord(
            id=f"usg-{len(self._records) + 1}",
            user_id=user_id,
            key_id=key_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
        )
        self._records.append(record)
        return record

    def get_summary(self, user_id: Optional[str] = None) -> Dict:
        records = (
            [r for r in self._records if r.user_id == user_id]
            if user_id
            else self._records
        )
        if not records:
            return {
                "total_requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "avg_ttft_ms": 0.0,
                "avg_latency_ms": 0.0,
            }

        total_reqs = len(records)
        prompt_toks = sum(r.prompt_tokens for r in records)
        comp_toks = sum(r.completion_tokens for r in records)
        avg_ttft = sum(r.ttft_ms for r in records) / total_reqs
        avg_lat = sum(r.total_latency_ms for r in records) / total_reqs

        return {
            "total_requests": total_reqs,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "total_tokens": prompt_toks + comp_toks,
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_latency_ms": round(avg_lat, 2),
        }


_usage_tracker_instance: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    global _usage_tracker_instance
    if _usage_tracker_instance is None:
        _usage_tracker_instance = UsageTracker()
    return _usage_tracker_instance
