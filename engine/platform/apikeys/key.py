import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class APIKey:
    id: str
    user_id: str
    name: str
    key_hash: str
    prefix: str  # First 12 chars e.g. sk-chatii-ab12
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    expires_at: Optional[float] = None
    is_active: bool = True
    rate_limit_rpm: int = 60
    total_requests: int = 0
    total_tokens: int = 0


class APIKeyManager:
    """Manages developer API keys with hashing, revocation, and stats."""

    def __init__(self) -> None:
        self._keys: Dict[str, APIKey] = {}
        self._hash_map: Dict[str, str] = {}  # key_hash -> key_id

    def create_key(
        self,
        user_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
        rate_limit_rpm: int = 60,
    ) -> tuple[APIKey, str]:
        raw_token = f"sk-chatii-{secrets.token_urlsafe(24)}"
        key_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        key_id = f"key-{uuid.uuid4().hex[:12]}"
        expires_at = (
            time.time() + (expires_in_days * 86400) if expires_in_days else None
        )

        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name.strip(),
            key_hash=key_hash,
            prefix=raw_token[:14] + "...",
            expires_at=expires_at,
            rate_limit_rpm=rate_limit_rpm,
        )

        self._keys[key_id] = api_key
        self._hash_map[key_hash] = key_id
        return api_key, raw_token

    def verify_key(self, raw_token: str) -> Optional[APIKey]:
        key_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        key_id = self._hash_map.get(key_hash)
        if not key_id:
            return None
        api_key = self._keys.get(key_id)
        if not api_key or not api_key.is_active:
            return None
        if api_key.expires_at and api_key.expires_at < time.time():
            return None

        # Update stats
        api_key.last_used_at = time.time()
        api_key.total_requests += 1
        return api_key

    def list_user_keys(self, user_id: str) -> List[APIKey]:
        return [k for k in self._keys.values() if k.user_id == user_id]

    def revoke_key(self, key_id: str, user_id: str) -> bool:
        key = self._keys.get(key_id)
        if key and key.user_id == user_id:
            key.is_active = False
            return True
        return False


_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
