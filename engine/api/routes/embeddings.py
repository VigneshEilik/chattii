from fastapi import APIRouter, Request

from engine.api.schema.openai import EmbeddingData, EmbeddingRequest, EmbeddingResponse
from engine.api.services.factory import ApiServiceFactory

router = APIRouter()


@router.post("/embeddings")
def create_embeddings(request: Request, body: EmbeddingRequest):
    state = request.app.state
    embedding_service = ApiServiceFactory.get_embedding_service(state)

    inputs = [body.input] if isinstance(body.input, str) else body.input
    data_list = []

    for idx, inp in enumerate(inputs):
        vec = embedding_service.generate_embeddings(body.model, inp)
        data_list.append(EmbeddingData(embedding=vec, index=idx))

    return EmbeddingResponse(
        data=data_list,
        model=body.model,
        usage={"prompt_tokens": len(inputs) * 4, "total_tokens": len(inputs) * 4},
    )
