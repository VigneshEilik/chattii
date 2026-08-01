from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from engine.api.schema.error import ErrorDetail, ErrorResponse


class ExceptionMiddleware(BaseHTTPMiddleware):
    """Maps uncaught backend code exceptions to unified error payloads."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except ValueError as val_err:
            err = ErrorResponse(
                error=ErrorDetail(
                    message=str(val_err),
                    type="invalid_request_error",
                    code="bad_request",
                    request_id=getattr(request.state, "request_id", None),
                )
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST, content=err.model_dump()
            )
        except Exception as exc:
            err = ErrorResponse(
                error=ErrorDetail(
                    message=f"An unexpected internal error occurred: {str(exc)}",
                    type="internal_server_error",
                    code="internal_error",
                    request_id=getattr(request.state, "request_id", None),
                )
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=err.model_dump(),
            )
