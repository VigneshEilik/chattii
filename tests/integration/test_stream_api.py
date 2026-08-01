from fastapi.testclient import TestClient

from engine.api.api import app


def test_stream_api_endpoint():
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer chatii-admin-root-key"}
        payload = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 16,
            "stream": True,
        }

        # Stream requests
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Verify stream content has data prefixes
        lines = response.text.split("\n")
        data_lines = [line for line in lines if line.startswith("data:")]
        assert len(data_lines) > 0
