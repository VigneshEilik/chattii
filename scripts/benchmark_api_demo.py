from engine.api.benchmark.api_benchmark import run_api_benchmark


def main() -> None:
    print("Initiating API Serving Gateway benchmark...")
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Authorization": "Bearer chatii-admin-root-key"}
    payload = {
        "model": "chatii-v1",
        "messages": [{"role": "user", "content": "Explain attention mechanism"}],
        "max_tokens": 16,
    }

    try:
        # Run 5 requests and print results
        res = run_api_benchmark(url, headers, payload, count=5)
        print("API Benchmark Results:")
        print(f"  Average Latency: {res['avg_latency']:.3f} seconds")
        print(f"  Throughput: {res['tps']:.2f} requests/sec")
    except Exception as e:
        print(f"Skipping API benchmark (server not running): {e}")


if __name__ == "__main__":
    main()
