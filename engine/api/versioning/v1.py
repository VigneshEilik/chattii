from fastapi import APIRouter

from engine.api.routes.admin import router as admin_router
from engine.api.routes.batches import router as batches_router
from engine.api.routes.chat import router as chat_router
from engine.api.routes.completion import router as completion_router
from engine.api.routes.embeddings import router as embeddings_router
from engine.api.routes.models import router as models_router
from engine.api.routes.tokenizer import router as tokenizer_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(completion_router)
router.include_router(embeddings_router)
router.include_router(models_router)
router.include_router(tokenizer_router)
router.include_router(admin_router)
router.include_router(batches_router)
