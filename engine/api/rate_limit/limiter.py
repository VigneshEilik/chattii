import time
from typing import Dict, List


class TokenBucket:
    """Token Bucket rate limiting algorithm."""

    def __init__(self, capacity: int, fill_rate: float) -> None:
        self.capacity = capacity
        self.fill_rate = fill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_update = time.time()

    def allow(self, tokens_requested: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Add new tokens
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.fill_rate)

        if self.tokens >= tokens_requested:
            self.tokens -= tokens_requested
            return True
        return False


class SlidingWindow:
    """Sliding Window rate limiting algorithm."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.timestamps: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        # Evict timestamps outside the sliding window
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

        if len(self.timestamps) < self.limit:
            self.timestamps.append(now)
            return True
        return False


class RateLimiter:
    """Manages rate limit buckets/windows mapped by client identifier key."""

    def __init__(self, config_dict: dict) -> None:
        self.config_dict = config_dict
        self.buckets: Dict[str, TokenBucket] = {}
        self.windows: Dict[str, SlidingWindow] = {}

        limit_cfg = config_dict.get("limits", {}).get("rate_limit", {})
        self.rpm = limit_cfg.get("requests_per_minute", 60)
        self.burst = limit_cfg.get("burst", 10)

    def is_allowed(self, identifier: str, use_sliding_window: bool = False) -> bool:
        if use_sliding_window:
            if identifier not in self.windows:
                self.windows[identifier] = SlidingWindow(
                    limit=self.rpm, window_seconds=60
                )
            return self.windows[identifier].allow()
        else:
            if identifier not in self.buckets:
                # Capacity = burst, fill rate = rpm / 60
                fill_rate = self.rpm / 60.0
                self.buckets[identifier] = TokenBucket(
                    capacity=self.burst, fill_rate=fill_rate
                )
            return self.buckets[identifier].allow()
