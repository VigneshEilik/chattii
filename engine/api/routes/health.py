from fastapi import APIRouter, Request, Response, status

router = APIRouter()


@router.get("/health/live")
def check_liveness():
    """Liveness probe: verifies that the server process is alive."""
    return {"status": "alive"}


@router.get("/health/ready")
def check_readiness(request: Request, response: Response):
    """Readiness probe: verifies that at least one model is successfully loaded and ready for serving."""
    state = request.app.state
    if not state.model_manager or not state.model_manager.loaded_models:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready", "reason": "No models loaded"}
    return {"status": "ready"}


@router.get("/health/startup")
def check_startup(request: Request, response: Response):
    """Startup probe: verifies that initialization, configuration, and dependencies have booted."""
    state = request.app.state
    if not state.config or not state.model_manager:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "starting"}
    return {"status": "started"}


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Favicon endpoint preventing browser 404/401 console logs."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/.well-known/{rest_of_path:path}", include_in_schema=False)
def well_known(rest_of_path: str):
    """Handles browser devtools and well-known requests cleanly."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)
