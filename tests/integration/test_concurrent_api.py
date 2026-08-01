import concurrent.futures

from fastapi.testclient import TestClient

from engine.api.api import app


def send_request(client):
    headers = {"Authorization": "Bearer chatii-admin-root-key"}
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }
    resp = client.post("/v1/chat/completions", json=payload, headers=headers)
    return resp.status_code


def test_concurrent_api_requests():
    with TestClient(app) as client:
        # Run 4 requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(send_request, client) for _ in range(4)]
            results = [f.result() for f in futures]

        assert all(status == 200 for status in results)
