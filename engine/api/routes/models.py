from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/models")
def list_models(request: Request):
    state = request.app.state
    # Query model registry
    models_metadata = state.model_manager.registry.list_models()
    return {"object": "list", "data": [m.to_dict() for m in models_metadata]}
