import requests


def main() -> None:
    print("Testing OpenAI Library API Compatibility...")
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Authorization": "Bearer chatii-admin-root-key"}

    # Mirror OpenAI payload structure exactly
    payload = {
        "model": "chatii-v1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.7,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        print("Status Code:", resp.status_code)
        print("Response JSON:")
        print(resp.json())
    except Exception as e:
        print(f"Skipping OpenAI compatibility check (API server not running): {e}")


if __name__ == "__main__":
    main()
