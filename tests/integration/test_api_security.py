from fastapi.testclient import TestClient

from engine.api.api import app


def test_missing_auth_header():
    with TestClient(app) as client:
        payload = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "ping"}],
        }
        # No auth header
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 401
        assert "error" in response.json()


def test_invalid_jwt_signature():
    with TestClient(app) as client:
        payload = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "ping"}],
        }
        # Malformed JWT token
        headers = {"Authorization": "Bearer headers.payloads.signature"}
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 401


def test_rate_limiting_security():
    with TestClient(app) as client:
        # Rapid requests to trigger 429
        headers = {"Authorization": "Bearer chatii-development-test-key"}
        payload = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
        }

        status_codes = []
        for _ in range(12):
            resp = client.post("/v1/chat/completions", json=payload, headers=headers)
            status_codes.append(resp.status_code)

        assert 429 in status_codes
