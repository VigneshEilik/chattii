import time
from typing import Any, Dict

import requests


def run_api_benchmark(
    url: str, headers: Dict[str, str], payload: Dict[str, Any], count: int = 10
) -> Dict[str, float]:
    """Generates benchmark HTTP completions request calls and compiles latency metrics."""
    latencies = []

    for _ in range(count):
        start = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                latencies.append(time.time() - start)
        except Exception:
            pass

    if not latencies:
        return {"avg_latency": 0.0, "tps": 0.0}

    avg = sum(latencies) / len(latencies)
    return {"avg_latency": avg, "tps": len(latencies) / sum(latencies)}
