"""Auth router — signup + login (issues a bearer token scoped to a tenant)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, Depends

from app.models.schemas import LoginRequest, SignupRequest, TokenResponse
from app.services.auth import create_token, user_store
from app.deps import Principal, get_principal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse)
async def signup(request: SignupRequest, response: Response) -> TokenResponse:
    try:
        user = await user_store.create_async(request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_token(user)
    response.set_cookie(key="vision_access_token", value=token, httponly=True, samesite="lax")
    return TokenResponse(access_token=token, tenant_id=user.tenant_id, email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, response: Response) -> TokenResponse:
    user = await user_store.verify_async(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = create_token(user)
    response.set_cookie(key="vision_access_token", value=token, httponly=True, samesite="lax")
    return TokenResponse(access_token=token, tenant_id=user.tenant_id, email=user.email)


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="vision_access_token", httponly=True, samesite="lax")
    return {"success": True}


@router.get("/me")
async def get_current_user(principal: Principal = Depends(get_principal)):
    return {
        "user_id": principal.user_id,
        "tenant_id": principal.tenant_id,
        "email": principal.email,
    }
