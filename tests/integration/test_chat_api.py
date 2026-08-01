from fastapi.testclient import TestClient

from engine.api.api import app


def test_chat_api_endpoint():
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer chatii-admin-root-key"}
        payload = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 16,
            "stream": False,
        }

        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
