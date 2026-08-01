from engine.api.auth.jwt import JwtAuthenticator
from engine.api.auth.key import KeyAuthenticator


def test_key_authenticator():
    auth = KeyAuthenticator(["root-key"])
    assert auth.authenticate("Bearer root-key") is True
    assert auth.authenticate("Bearer invalid-key") is False
    assert auth.authenticate("") is False


def test_jwt_authenticator():
    auth = JwtAuthenticator("my-secret")
    payload = {"sub": "user1", "role": "admin"}
    token = auth.encode(payload)

    decoded = auth.decode_and_verify(token)
    assert decoded is not None
    assert decoded["sub"] == "user1"

    # Invalid token check
    assert auth.decode_and_verify("invalid.token.here") is None
