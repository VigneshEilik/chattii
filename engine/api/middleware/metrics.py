import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """Tracks latency metrics for active request pipelines."""

    async def dispatch(self, request: Request, call_next):
        tracker = request.app.state.metrics_tracker
        if tracker:
            tracker.increment_active_requests()

        start_time = time.time()
        try:
            response = await call_next(request)
            return response
        finally:
            latency = time.time() - start_time
            if tracker:
                tracker.decrement_active_requests()
                tracker.record_request_latency(latency)
