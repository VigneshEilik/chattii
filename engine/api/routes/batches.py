import time
import uuid

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

router = APIRouter()


class BatchRequestPayload(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str


@router.post("/batches", status_code=status.HTTP_201_CREATED)
def create_batch_job(request: Request, body: BatchRequestPayload):
    batch_id = f"batch_{uuid.uuid4()}"
    return {
        "id": batch_id,
        "object": "batch",
        "endpoint": body.endpoint,
        "errors": None,
        "input_file_id": body.input_file_id,
        "completion_window": body.completion_window,
        "status": "validating",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": int(time.time()),
        "in_progress_at": None,
        "expires_at": int(time.time()) + 86400,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "metadata": None,
    }
