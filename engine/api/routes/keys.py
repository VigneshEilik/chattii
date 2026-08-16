from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.platform.apikeys.key import get_api_key_manager

router = APIRouter(prefix="/v1/keys", tags=["API Keys"])


class CreateKeyRequest(BaseModel):
    user_id: str = "usr-default"
    name: str
    expires_in_days: Optional[int] = None
    rate_limit_rpm: int = 60


@router.post("")
def create_api_key(req: CreateKeyRequest):
    mgr = get_api_key_manager()
    api_key, raw_token = mgr.create_key(
        user_id=req.user_id,
        name=req.name,
        expires_in_days=req.expires_in_days,
        rate_limit_rpm=req.rate_limit_rpm,
    )
    return {
        "id": api_key.id,
        "name": api_key.name,
        "prefix": api_key.prefix,
        "raw_key": raw_token,  # Shown only once
        "created_at": api_key.created_at,
        "expires_at": api_key.expires_at,
        "rate_limit_rpm": api_key.rate_limit_rpm,
    }


@router.get("")
def list_api_keys(user_id: str = "usr-default"):
    mgr = get_api_key_manager()
    keys = mgr.list_user_keys(user_id)
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.prefix,
            "created_at": k.created_at,
            "last_used_at": k.last_used_at,
            "expires_at": k.expires_at,
            "is_active": k.is_active,
            "total_requests": k.total_requests,
        }
        for k in keys
    ]


@router.delete("/{key_id}")
def revoke_api_key(key_id: str, user_id: str = "usr-default"):
    mgr = get_api_key_manager()
    success = mgr.revoke_key(key_id, user_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="API key not found or not owned by user."
        )
    return {"status": "revoked", "id": key_id}
