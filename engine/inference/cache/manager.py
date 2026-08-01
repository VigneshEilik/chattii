from typing import Dict, Optional

from engine.inference.cache.kv_cache import BaseCache, KVCache


class CacheManager:
    """Coordinates allocation, reuse, and eviction policies across active sessions."""

    def __init__(self, max_cached_sessions: int = 32) -> None:
        self.max_cached_sessions = max_cached_sessions
        self.cached_sessions: Dict[str, BaseCache] = {}

    def allocate(self, session_id: str, num_layers: int = 12) -> BaseCache:
        if len(self.cached_sessions) >= self.max_cached_sessions:
            # Evict first session (FIFO eviction policy)
            oldest_id = next(iter(self.cached_sessions))
            self.free(oldest_id)

        cache = KVCache(num_layers=num_layers)
        self.cached_sessions[session_id] = cache
        return cache

    def get(self, session_id: str) -> Optional[BaseCache]:
        return self.cached_sessions.get(session_id)

    def free(self, session_id: str) -> None:
        if session_id in self.cached_sessions:
            cache = self.cached_sessions.pop(session_id)
            cache.reset()

    def clear(self) -> None:
        for cache in self.cached_sessions.values():
            cache.reset()
        self.cached_sessions.clear()
