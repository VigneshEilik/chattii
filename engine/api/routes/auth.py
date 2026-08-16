from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from engine.platform.auth.session import get_session_manager
from engine.platform.auth.user import get_user_repository

router = APIRouter(prefix="/v1/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/register")
def register(req: RegisterRequest):
    repo = get_user_repository()
    try:
        user = repo.create_user(
            email=req.email,
            username=req.username,
            password=req.password,
        )
        session = get_session_manager().create_session(user)
        return {
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role,
            },
            "access_token": session.token,
            "refresh_token": session.refresh_token,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
def login(req: LoginRequest):
    repo = get_user_repository()
    user = repo.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    session = get_session_manager().create_session(user)
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role,
        },
        "access_token": session.token,
        "refresh_token": session.refresh_token,
    }


@router.post("/refresh")
def refresh(req: RefreshRequest):
    session = get_session_manager().refresh_session(req.refresh_token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")
    return {
        "access_token": session.token,
    }


@router.get("/me")
def get_me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid authorization token."
        )
    token = authorization.split(" ")[1]
    session = get_session_manager().get_session_by_token(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    user = get_user_repository().get_by_id(session.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "role": user.role,
        "created_at": user.created_at,
    }
