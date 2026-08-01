import concurrent.futures

from fastapi.testclient import TestClient

from engine.api.api import app


def run_single_client(client, prompt: str):
    headers = {"Authorization": "Bearer chatii-admin-root-key"}
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
    }
    resp = client.post("/v1/chat/completions", json=payload, headers=headers)
    return resp.status_code


def test_api_load_capacity():
    with TestClient(app) as client:
        # Run 10 requests concurrently representing 10 clients
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(run_single_client, client, f"Query {i}")
                for i in range(10)
            ]
            results = [f.result() for f in futures]

        assert all(status == 200 for status in results)
