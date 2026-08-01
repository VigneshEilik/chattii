from sdk.python.chatii_sdk.client import ChatiiClient


def main() -> None:
    print("Initializing Chatii Client...")
    # Initialize pointing to a mock/running endpoint with standard auth keys
    client = ChatiiClient(
        base_url="http://localhost:8000/v1", api_key="chatii-admin-root-key"
    )

    print("\nListing available models:")
    try:
        models = client.models.list()
        print(models)
    except Exception as e:
        print(f"Skipping models list (API server not running): {e}")

    print("\nSending a chat completion request:")
    try:
        messages = [{"role": "user", "content": "What is self-attention?"}]
        response = client.chat.create(
            model="chatii-v1", messages=messages, max_tokens=32
        )
        print("Response:", response)
    except Exception as e:
        print(f"Skipping completions check (API server not running): {e}")


if __name__ == "__main__":
    main()
