import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from engine.api.schema.openai import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    Usage,
)
from engine.api.services.factory import ApiServiceFactory
from engine.inference.config import GenerationConfig

router = APIRouter()


@router.post("/chat/completions")
def create_chat_completion(request: Request, body: ChatRequest):
    state = request.app.state
    chat_service = ApiServiceFactory.get_chat_service(state)

    config = GenerationConfig(
        config_dict={
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_new_tokens": body.max_tokens,
            "presence_penalty": body.presence_penalty,
            "frequency_penalty": body.frequency_penalty,
        }
    )

    if body.stream:
        # Stream response using Server-Sent Events (SSE)
        gen = chat_service.stream_chat(
            model_id=body.model,
            messages=[{"role": m.role, "content": m.content} for m in body.messages],
            config=config,
            session_id=str(uuid.uuid4()),
        )

        from engine.api.streaming.sse import format_sse_stream

        return StreamingResponse(
            format_sse_stream(body.model, gen), media_type="text/event-stream"
        )

    # Standard JSON Response
    response = chat_service.generate_chat(
        model_id=body.model,
        messages=[{"role": m.role, "content": m.content} for m in body.messages],
        config=config,
        session_id=str(uuid.uuid4()),
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response.text),
                finish_reason=response.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.generated_tokens,
            total_tokens=response.prompt_tokens + response.generated_tokens,
        ),
    )
