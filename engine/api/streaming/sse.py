import json
import time
import uuid
from typing import Iterator


def format_sse_stream(model_id: str, token_generator: Iterator[str]) -> Iterator[str]:
    """Formats string tokens into standard Server-Sent Events (SSE) events."""
    chunk_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    for _idx, token in enumerate(token_generator):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final stop chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "eos_token",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
