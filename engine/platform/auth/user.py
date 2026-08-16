import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from engine.platform.auth.password import hash_password, verify_password


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    DEVELOPER = "developer"


@dataclass
class User:
    id: str
    email: str
    username: str
    password_hash: str
    role: UserRole = UserRole.USER
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)


class UserRepository:
    """Thread-safe persistent/in-memory user store."""

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        self._email_index: Dict[str, str] = {}
        # Seed default admin user
        self.create_user(
            email="admin@chatii.ai",
            username="admin",
            password="chatii-admin-password",
            role=UserRole.ADMIN,
        )

    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
    ) -> User:
        email_clean = email.lower().strip()
        if email_clean in self._email_index:
            raise ValueError(f"User with email '{email}' already exists.")

        user_id = f"usr-{uuid.uuid4().hex[:12]}"
        user = User(
            id=user_id,
            email=email_clean,
            username=username.strip(),
            password_hash=hash_password(password),
            role=role,
        )
        self._users[user_id] = user
        self._email_index[email_clean] = user_id
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        user_id = self._email_index.get(email.lower().strip())
        if user_id:
            return self._users.get(user_id)
        return None

    def authenticate(self, email: str, password: str) -> Optional[User]:
        user = self.get_by_email(email)
        if user and verify_password(password, user.password_hash) and user.is_active:
            return user
        return None

    def list_users(self) -> List[User]:
        return list(self._users.values())

    def delete_user(self, user_id: str) -> bool:
        user = self._users.pop(user_id, None)
        if user:
            self._email_index.pop(user.email, None)
            return True
        return False


_user_repo_instance: Optional[UserRepository] = None


def get_user_repository() -> UserRepository:
    global _user_repo_instance
    if _user_repo_instance is None:
        _user_repo_instance = UserRepository()
    return _user_repo_instance
