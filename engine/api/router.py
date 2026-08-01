from fastapi import APIRouter

from engine.api.routes.health import router as health_router
from engine.api.routes.metrics import router as metrics_router
from engine.api.streaming.websocket import router as ws_router
from engine.api.versioning.v1 import router as v1_router

router = APIRouter()

# Include version v1 router
router.include_router(v1_router, prefix="/v1")

# Include ws routes
router.include_router(ws_router)

# Include health & metrics endpoints at root level
router.include_router(health_router)
router.include_router(metrics_router)
