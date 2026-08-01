from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from engine.api.auth.jwt import JwtAuthenticator
from engine.api.auth.key import KeyAuthenticator


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforces API Bearer Key / JWT token checks on non-public route endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip authorization for public health, metrics, and API document interfaces
        public_paths = [
            "/health/live",
            "/health/ready",
            "/health/startup",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        if (
            path in public_paths
            or path.startswith("/docs")
            or path.startswith("/redoc")
        ):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        config = request.app.state.config

        # Auth Config Retrieval
        api_keys = config.get("auth", {}).get("api_keys", [])
        jwt_sec = config.get("auth", {}).get("jwt", {}).get("secret_key", "")

        key_authenticator = KeyAuthenticator(api_keys)
        jwt_authenticator = JwtAuthenticator(jwt_sec)

        # Allow if static API key or JWT token is valid
        is_key_valid = key_authenticator.authenticate(auth_header)
        is_jwt_valid = jwt_authenticator.authenticate(auth_header) is not None

        if not is_key_valid and not is_jwt_valid:
            from fastapi.responses import JSONResponse

            from engine.api.schema.error import ErrorDetail, ErrorResponse

            err = ErrorResponse(
                error=ErrorDetail(
                    message="Missing or invalid authentication credentials.",
                    type="invalid_request_error",
                    code="unauthorized",
                    request_id=getattr(request.state, "request_id", None),
                )
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED, content=err.model_dump()
            )

        return await call_next(request)
