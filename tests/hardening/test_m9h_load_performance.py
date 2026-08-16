import concurrent.futures
import time

from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def _send_concurrent_req(client, idx: int):
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": f"Load query {idx}"}],
        "max_tokens": 10,
    }
    t0 = time.time()
    resp = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
    t1 = time.time()
    return resp.status_code, (t1 - t0)


def test_concurrent_load_and_ttft_bounds():
    with TestClient(app) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(_send_concurrent_req, client, i) for i in range(5)
            ]
            results = [f.result() for f in futures]

        statuses = [r[0] for r in results]
        latencies = [r[1] for r in results]

        assert all(s == 200 for s in statuses)
        # Average latency should be within reasonable bound (< 3.0 seconds locally)
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 3.0
