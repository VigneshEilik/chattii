from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def test_invalid_json_request_body():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content="invalid json body text",
            headers={**AUTH_HEADERS, "Content-Type": "application/json"},
        )
        assert resp.status_code in [400, 422]


def test_nonexistent_route_404():
    with TestClient(app) as client:
        resp = client.get("/v1/nonexistent-path", headers=AUTH_HEADERS)
        assert resp.status_code == 404


def test_missing_required_message_field():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "chatii-v1"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 422
