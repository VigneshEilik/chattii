from typing import Any, Dict


class PromptPrefixCache:
    """Caches computed prompt prefix key-value matrices to optimize TTFT."""

    def __init__(self) -> None:
        self.prefixes: Dict[str, Any] = {}

    def get(self, prompt: str) -> Any:
        return self.prefixes.get(prompt)

    def set(self, prompt: str, cache_value: Any) -> None:
        self.prefixes[prompt] = cache_value

    def clear(self) -> None:
        self.prefixes.clear()


class SessionPrefixCache:
    """Caches prefix token key-value states mapped to specific user sessions."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Any] = {}

    def get(self, session_id: str) -> Any:
        return self.sessions.get(session_id)

    def set(self, session_id: str, cache_value: Any) -> None:
        self.sessions[session_id] = cache_value

    def clear(self) -> None:
        self.sessions.clear()


class GlobalPrefixCache:
    """Caches global systems messages prefix tensors (e.g. system instructions)."""

    def __init__(self) -> None:
        self.global_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.global_cache.get(key)

    def set(self, key: str, cache_value: Any) -> None:
        self.global_cache[key] = cache_value

    def clear(self) -> None:
        self.global_cache.clear()
