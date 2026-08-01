from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Protects server allocations by enforcing client requests rate limit ceilings."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Skip health/metrics/docs from rate limiting
        if path.startswith("/health") or path == "/metrics" or path.startswith("/docs"):
            return await call_next(request)

        limiter = request.app.state.limiter
        # Rate limit identifier: prioritize authorization token/API key, fallback to client IP
        auth_header = request.headers.get("Authorization", "")
        identifier = (
            auth_header
            if auth_header
            else (request.client.host if request.client else "unknown")
        )

        if limiter and not limiter.is_allowed(identifier):
            from fastapi.responses import JSONResponse

            from engine.api.schema.error import ErrorDetail, ErrorResponse

            err = ErrorResponse(
                error=ErrorDetail(
                    message="Rate limit exceeded. Too many requests.",
                    type="rate_limit_error",
                    code="rate_limit_exceeded",
                    request_id=getattr(request.state, "request_id", None),
                )
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=err.model_dump()
            )

        return await call_next(request)
