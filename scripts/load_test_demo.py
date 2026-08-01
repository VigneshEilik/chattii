import concurrent.futures
import time

import requests


def worker(thread_idx: int) -> int:
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Authorization": "Bearer chatii-admin-root-key"}
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": f"Ping from worker {thread_idx}"}],
        "max_tokens": 8,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        return resp.status_code
    except Exception:
        return 500


def main() -> None:
    print("Initiating Chatii Load Test Simulator...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        results = [f.result() for f in futures]

    duration = time.time() - start_time
    print(f"Load test simulation completed in {duration:.3f} seconds.")
    print(f"Results status codes: {results}")


if __name__ == "__main__":
    main()
