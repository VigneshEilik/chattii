from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel

from engine.api.services.factory import ApiServiceFactory

router = APIRouter()


class TokenizeRequest(BaseModel):
    model: str
    text: str


class DetokenizeRequest(BaseModel):
    model: str
    tokens: List[int]


@router.post("/tokenize")
def tokenize_text(request: Request, body: TokenizeRequest):
    state = request.app.state
    tokenizer_service = ApiServiceFactory.get_tokenizer_service(state)
    token_ids = tokenizer_service.encode(body.model, body.text)
    return {"tokens": token_ids, "length": len(token_ids)}


@router.post("/detokenize")
def detokenize_ids(request: Request, body: DetokenizeRequest):
    state = request.app.state
    tokenizer_service = ApiServiceFactory.get_tokenizer_service(state)
    text = tokenizer_service.decode(body.model, body.tokens)
    return {"text": text}
