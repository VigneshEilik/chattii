import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured access logger outputting metrics in JSON format."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        latency = time.time() - start_time
        request_id = getattr(request.state, "request_id", "unknown")

        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_seconds": latency,
        }

        # Access logger
        logger = logging.getLogger("chatii.access")
        logger.info(log_data)

        return response
