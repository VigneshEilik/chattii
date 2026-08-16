from fastapi import APIRouter

from engine.api.routes.admin import router as admin_router
from engine.api.routes.auth import router as auth_router
from engine.api.routes.conversations import router as conv_router
from engine.api.routes.health import router as health_router
from engine.api.routes.keys import router as keys_router
from engine.api.routes.metrics import router as metrics_router
from engine.api.routes.multimodal import router as mm_router
from engine.api.routes.rag import router as rag_router
from engine.api.routes.usage import router as usage_router
from engine.api.streaming.websocket import router as ws_router
from engine.api.versioning.v1 import router as v1_router

router = APIRouter()

# Include version v1 router
router.include_router(v1_router, prefix="/v1")

# Include M9 feature routers
router.include_router(auth_router)
router.include_router(conv_router)
router.include_router(keys_router)
router.include_router(usage_router)
router.include_router(rag_router)
router.include_router(mm_router)
router.include_router(admin_router)

# Include ws routes
router.include_router(ws_router)

# Include health & metrics endpoints at root level
router.include_router(health_router)
router.include_router(metrics_router)
