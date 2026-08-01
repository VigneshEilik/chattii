import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.api.services.factory import ApiServiceFactory
from engine.api.streaming.connection_manager import ConnectionManager
from engine.inference.config import GenerationConfig

router = APIRouter()
manager = ConnectionManager()


@router.websocket("/v1/chat/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    state = websocket.app.state
    ApiServiceFactory.get_chat_service(state)

    try:
        while True:
            # Expecting JSON request: {"model": "chatii-v1", "prompt": "...", "max_tokens": 64}
            data = await websocket.receive_text()
            payload = json.loads(data)

            model_id = payload.get("model", "chatii-v1")
            prompt = payload.get("prompt", "")
            max_tokens = payload.get("max_tokens", 64)

            config = GenerationConfig(config_dict={"max_new_tokens": max_tokens})

            # Simple direct generation stream
            service = state.model_manager.get_model(model_id)
            if not service:
                await websocket.send_text(
                    json.dumps({"error": f"Model '{model_id}' not found"})
                )
                continue

            for token in service.stream(prompt, config=config):
                await websocket.send_text(json.dumps({"token": token}))

            await websocket.send_text(json.dumps({"status": "done"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
