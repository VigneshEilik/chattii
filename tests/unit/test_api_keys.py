from engine.platform.apikeys.key import APIKeyManager


def test_create_api_key():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-1", "Prod Key")
    assert key.id.startswith("key-")
    assert key.name == "Prod Key"
    assert raw_token.startswith("sk-chatii-")


def test_verify_valid_api_key():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-2", "Test Key")
    verified = mgr.verify_key(raw_token)
    assert verified is not None
    assert verified.id == key.id
    assert verified.total_requests == 1


def test_verify_invalid_api_key():
    mgr = APIKeyManager()
    assert mgr.verify_key("sk-chatii-invalid-key-token") is None


def test_list_user_keys():
    mgr = APIKeyManager()
    k1, _ = mgr.create_key("usr-3", "Key 1")
    k2, _ = mgr.create_key("usr-3", "Key 2")
    mgr.create_key("usr-4", "Key 3")

    user_keys = mgr.list_user_keys("usr-3")
    assert len(user_keys) == 2
    assert {k.id for k in user_keys} == {k1.id, k2.id}


def test_revoke_api_key():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-5", "Revoke Key")
    assert mgr.revoke_key(key.id, "usr-5") is True
    assert mgr.verify_key(raw_token) is None


def test_revoke_wrong_user_fails():
    mgr = APIKeyManager()
    key, _ = mgr.create_key("usr-6", "Owner Key")
    assert mgr.revoke_key(key.id, "usr-other") is False


def test_expired_key_verification():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-7", "Expired Key", expires_in_days=-1)
    assert mgr.verify_key(raw_token) is None


def test_key_prefix():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-8", "Prefix Key")
    assert key.prefix.startswith("sk-chatii-")


def test_last_used_at_updated():
    mgr = APIKeyManager()
    key, raw_token = mgr.create_key("usr-9", "Usage Key")
    assert key.last_used_at is None
    mgr.verify_key(raw_token)
    assert key.last_used_at is not None


def test_default_rate_limit():
    mgr = APIKeyManager()
    key, _ = mgr.create_key("usr-10", "Rate Limit Key", rate_limit_rpm=120)
    assert key.rate_limit_rpm == 120
