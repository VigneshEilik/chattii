import time
from typing import Dict, List


class SessionState:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: List[dict] = []
        self.last_accessed = time.time()


class SessionManager:
    """Orchestrates persistent session instances, TTL expirations, and garbage collection."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        self.cleanup_expired()
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        else:
            self.sessions[session_id].last_accessed = time.time()
        return self.sessions[session_id]

    def clear(self) -> None:
        self.sessions.clear()

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, state in self.sessions.items()
            if now - state.last_accessed > self.ttl_seconds
        ]
        for sid in expired:
            self.sessions.pop(sid)
