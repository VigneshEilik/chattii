import asyncio
import json

try:
    import websockets
except ImportError:
    websockets = None


async def run_client():
    if websockets is None:
        print(
            "Please install websockets package (pip install websockets) to run this demo."
        )
        return

    uri = "ws://localhost:8000/v1/chat/ws"
    print(f"Connecting to WebSocket serving gateway at {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            payload = {
                "model": "chatii-v1",
                "prompt": "Hello from WebSocket client!",
                "max_tokens": 32,
            }
            await websocket.send(json.dumps(payload))
            print("Payload sent. Awaiting tokens...")

            while True:
                response = await websocket.recv()
                data = json.loads(response)
                if data.get("status") == "done" or "error" in data:
                    print("\nStream finished.")
                    break
                print(data.get("token", ""), end="", flush=True)
    except Exception as e:
        print(f"Skipping WebSocket connection (server not running): {e}")


def main() -> None:
    asyncio.run(run_client())


if __name__ == "__main__":
    main()
