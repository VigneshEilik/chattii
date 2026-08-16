from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.platform.conversations.models import MessageRole
from engine.platform.conversations.repository import get_conversation_repository

router = APIRouter(prefix="/v1/conversations", tags=["Conversations"])


class CreateConversationRequest(BaseModel):
    user_id: str = "usr-default"
    title: str = "New Conversation"
    model: str = "chatii-v1"
    system_prompt: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    role: MessageRole
    content: str


@router.post("")
def create_conversation(req: CreateConversationRequest):
    repo = get_conversation_repository()
    conv = repo.create_conversation(
        user_id=req.user_id,
        title=req.title,
        model=req.model,
        system_prompt=req.system_prompt,
    )
    return conv


@router.get("")
def list_conversations(user_id: str = "usr-default"):
    repo = get_conversation_repository()
    return repo.list_user_conversations(user_id)


@router.get("/{conv_id}")
def get_conversation(conv_id: str):
    repo = get_conversation_repository()
    conv = repo.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = repo.get_messages(conv_id)
    return {
        "conversation": conv,
        "messages": messages,
    }


@router.put("/{conv_id}")
def rename_conversation(conv_id: str, req: RenameConversationRequest):
    repo = get_conversation_repository()
    conv = repo.rename_conversation(conv_id, req.title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str):
    repo = get_conversation_repository()
    success = repo.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "id": conv_id}


@router.post("/{conv_id}/messages")
def add_message(conv_id: str, req: AddMessageRequest):
    repo = get_conversation_repository()
    try:
        msg = repo.add_message(conv_id=conv_id, role=req.role, content=req.content)
        return msg
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
