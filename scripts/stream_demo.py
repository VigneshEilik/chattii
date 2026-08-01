import sys

from sdk.python.chatii_sdk.client import ChatiiClient


def main() -> None:
    print("Initializing Chatii Streaming Client...")
    client = ChatiiClient(
        base_url="http://localhost:8000/v1", api_key="chatii-admin-root-key"
    )

    print("\nRequesting token stream:")
    try:
        messages = [{"role": "user", "content": "Explain rotary embeddings."}]
        stream = client.chat.create(
            model="chatii-v1", messages=messages, max_tokens=64, stream=True
        )

        print("Stream: ", end="", flush=True)
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content", "")
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
    except Exception as e:
        print(f"Skipping streaming check (API server not running): {e}")


if __name__ == "__main__":
    main()
