import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from engine.api.schema.openai import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from engine.api.services.factory import ApiServiceFactory
from engine.inference.config import GenerationConfig

router = APIRouter()


@router.post("/completions")
def create_completion(request: Request, body: CompletionRequest):
    state = request.app.state
    completion_service = ApiServiceFactory.get_completion_service(state)

    config = GenerationConfig(
        config_dict={
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_new_tokens": body.max_tokens,
        }
    )

    prompt_str = body.prompt if isinstance(body.prompt, str) else "".join(body.prompt)

    if body.stream:
        gen = completion_service.stream_completion(
            model_id=body.model,
            prompt=prompt_str,
            config=config,
            session_id=str(uuid.uuid4()),
        )

        from engine.api.streaming.sse import format_sse_stream

        return StreamingResponse(
            format_sse_stream(body.model, gen), media_type="text/event-stream"
        )

    response = completion_service.generate_completion(
        model_id=body.model,
        prompt=prompt_str,
        config=config,
        session_id=str(uuid.uuid4()),
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=body.model,
        choices=[
            CompletionChoice(
                text=response.text, index=0, finish_reason=response.finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.generated_tokens,
            total_tokens=response.prompt_tokens + response.generated_tokens,
        ),
    )
