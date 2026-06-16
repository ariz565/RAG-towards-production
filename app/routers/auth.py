"""Auth router — signup + login (issues a bearer token scoped to a tenant)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import LoginRequest, SignupRequest, TokenResponse
from app.services.auth import create_token, user_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=TokenResponse)
async def signup(request: SignupRequest) -> TokenResponse:
    try:
        user = user_store.create(request.email, request.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TokenResponse(access_token=create_token(user), tenant_id=user.tenant_id, email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = user_store.verify(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="invalid email or password")
    return TokenResponse(access_token=create_token(user), tenant_id=user.tenant_id, email=user.email)
