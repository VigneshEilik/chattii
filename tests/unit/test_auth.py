import pytest

from engine.platform.auth.password import hash_password, verify_password
from engine.platform.auth.session import SessionManager
from engine.platform.auth.user import UserRepository, UserRole


def test_password_hashing():
    pw = "secret-pass-123"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed) is True
    assert verify_password("wrong-pass", hashed) is False


def test_password_verify_invalid_hash():
    assert verify_password("pass", "malformed_hash") is False


def test_user_repository_creation():
    repo = UserRepository()
    user = repo.create_user("user1@test.com", "user1", "pass123")
    assert user.id.startswith("usr-")
    assert user.email == "user1@test.com"
    assert user.role == UserRole.USER


def test_user_repository_duplicate_email():
    repo = UserRepository()
    repo.create_user("user2@test.com", "user2", "pass123")
    with pytest.raises(ValueError):
        repo.create_user("user2@test.com", "user2_diff", "pass456")


def test_user_repository_get_by_id():
    repo = UserRepository()
    user = repo.create_user("user3@test.com", "user3", "pass123")
    fetched = repo.get_by_id(user.id)
    assert fetched == user


def test_user_repository_get_by_email():
    repo = UserRepository()
    user = repo.create_user("user4@test.com", "user4", "pass123")
    fetched = repo.get_by_email("USER4@test.com")
    assert fetched == user


def test_user_repository_authenticate_success():
    repo = UserRepository()
    repo.create_user("auth1@test.com", "auth1", "valid_password")
    user = repo.authenticate("auth1@test.com", "valid_password")
    assert user is not None
    assert user.email == "auth1@test.com"


def test_user_repository_authenticate_failure():
    repo = UserRepository()
    repo.create_user("auth2@test.com", "auth2", "valid_password")
    assert repo.authenticate("auth2@test.com", "wrong_password") is None
    assert repo.authenticate("nonexistent@test.com", "valid_password") is None


def test_user_repository_delete():
    repo = UserRepository()
    user = repo.create_user("del@test.com", "del", "pass")
    assert repo.delete_user(user.id) is True
    assert repo.get_by_id(user.id) is None


def test_session_manager_create_and_get():
    repo = UserRepository()
    user = repo.create_user("sess1@test.com", "sess1", "pass")
    sm = SessionManager()
    session = sm.create_session(user)
    assert session.user_id == user.id

    fetched = sm.get_session_by_token(session.token)
    assert fetched == session


def test_session_manager_refresh_token():
    repo = UserRepository()
    user = repo.create_user("sess2@test.com", "sess2", "pass")
    sm = SessionManager()
    session = sm.create_session(user)
    old_token = session.token

    refreshed = sm.refresh_session(session.refresh_token)
    assert refreshed is not None
    assert refreshed.token != old_token
    assert sm.get_session_by_token(old_token) is None


def test_session_manager_revoke():
    repo = UserRepository()
    user = repo.create_user("sess3@test.com", "sess3", "pass")
    sm = SessionManager()
    session = sm.create_session(user)
    assert sm.revoke_session(session.session_id) is True
    assert sm.get_session_by_token(session.token) is None
