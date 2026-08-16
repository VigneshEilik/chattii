import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from engine.platform.auth.user import User


@dataclass
class Session:
    session_id: str
    user_id: str
    token: str
    refresh_token: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600 * 24)  # 24h
    user_agent: str = "ChatiiClient"


class SessionManager:
    """Manages active user sessions and refresh tokens."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._token_map: Dict[str, str] = {}
        self._refresh_map: Dict[str, str] = {}

    def create_session(self, user: User, user_agent: str = "ChatiiClient") -> Session:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        access_token = f"chatii-acc-{uuid.uuid4().hex}"
        refresh_token = f"chatii-ref-{uuid.uuid4().hex}"

        session = Session(
            session_id=session_id,
            user_id=user.id,
            token=access_token,
            refresh_token=refresh_token,
            user_agent=user_agent,
        )
        self._sessions[session_id] = session
        self._token_map[access_token] = session_id
        self._refresh_map[refresh_token] = session_id
        return session

    def get_session_by_token(self, token: str) -> Optional[Session]:
        session_id = self._token_map.get(token)
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if session and session.expires_at > time.time():
            return session
        return None

    def refresh_session(self, refresh_token: str) -> Optional[Session]:
        session_id = self._refresh_map.get(refresh_token)
        if not session_id:
            return None
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Rotate tokens
        self._token_map.pop(session.token, None)
        session.token = f"chatii-acc-{uuid.uuid4().hex}"
        session.expires_at = time.time() + 3600 * 24
        self._token_map[session.token] = session.session_id
        return session

    def revoke_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session:
            self._token_map.pop(session.token, None)
            self._refresh_map.pop(session.refresh_token, None)
            return True
        return False


_session_manager_instance: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance
