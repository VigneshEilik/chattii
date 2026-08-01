from fastapi import APIRouter, Request, status
from pydantic import BaseModel

router = APIRouter()


class AdminModelPayload(BaseModel):
    model: str


@router.post("/admin/load", status_code=status.HTTP_202_ACCEPTED)
def admin_load_model(request: Request, body: AdminModelPayload):
    state = request.app.state
    state.model_manager.load_model(body.model)
    return {"status": "loaded", "model": body.model}


@router.post("/admin/unload", status_code=status.HTTP_202_ACCEPTED)
def admin_unload_model(request: Request, body: AdminModelPayload):
    state = request.app.state
    state.model_manager.unload_model(body.model)
    return {"status": "unloaded", "model": body.model}
