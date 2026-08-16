from typing import Optional

from fastapi import APIRouter

from engine.platform.usage.tracker import get_usage_tracker

router = APIRouter(prefix="/v1/usage", tags=["Usage"])


@router.get("/summary")
def get_usage_summary(user_id: Optional[str] = None):
    tracker = get_usage_tracker()
    return tracker.get_summary(user_id=user_id)
