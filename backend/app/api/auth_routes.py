"""
backend/app/api/auth_routes.py
==============================
Authentication API endpoints:
  - POST /auth/register : Register a new user account (with bcrypt hashing & duplicate detection)
  - POST /auth/login    : Authenticate credentials and return JWT bearer token
  - GET  /auth/me       : Retrieve current authenticated user profile
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    revoke_token,
    security_scheme,
    verify_password,
)
from backend.app.models.user import User
from backend.app.schemas.auth_schemas import (
    ChangePasswordRequest,
    MessageResponse,
    TokenResponse,
    UserLoginRequest,
    UserProfileUpdateRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.app.services import db_service

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Validates full name, email format, and password strength; hashes password with bcrypt; and returns a JWT access token.",
)
async def register_user(
    payload: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Create a new user account with duplicate email protection."""
    # Check if email is already registered
    existing_user = db_service.get_user_by_email(db, payload.email)
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address is already registered.",
        )

    # Hash password securely with bcrypt
    try:
        pw_hash = hash_password(payload.password)
    except Exception as exc:
        logger.error("Password hashing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to securely process password.",
        )

    # Persist user in PostgreSQL
    try:
        user = db_service.create_user(
            db=db,
            full_name=payload.full_name,
            email=payload.email,
            hashed_password=pw_hash,
        )
        db.commit()
        db.refresh(user)
    except Exception as exc:
        db.rollback()
        logger.error("User persistence failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user in database.",
        )

    # Generate JWT access token
    access_token = create_access_token(user_id=user.id, email=user.email)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


# Brute-force rate limiting tracker for login attempts
_LOGIN_ATTEMPTS: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60.0  # seconds
MAX_FAILED_ATTEMPTS = 5


def _check_login_rate_limit(key: str) -> None:
    now = time.time()
    attempts = [t for t in _LOGIN_ATTEMPTS[key] if now - t < RATE_LIMIT_WINDOW]
    _LOGIN_ATTEMPTS[key] = attempts
    if len(attempts) >= MAX_FAILED_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please wait 60 seconds before trying again.",
        )


def _record_failed_login(key: str) -> None:
    _LOGIN_ATTEMPTS[key].append(time.time())


def _reset_login_attempts(key: str) -> None:
    _LOGIN_ATTEMPTS.pop(key, None)


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login with email and password",
    description="Verifies user credentials against stored bcrypt hash, enforces rate-limiting, and returns a signed JWT access token.",
)
async def login_user(
    request: Request,
    payload: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """Authenticate credentials, protect against brute-force attacks, and issue JWT bearer token."""
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"{client_ip}:{payload.email.lower()}"
    _check_login_rate_limit(rate_key)

    user = db_service.get_user_by_email(db, payload.email)
    if user is None:
        _record_failed_login(rate_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(payload.password, user.hashed_password):
        _record_failed_login(rate_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _reset_login_attempts(rate_key)
    access_token = create_access_token(user_id=user.id, email=user.email)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@auth_router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke active JWT bearer token",
    description="Registers the active JWT token in PostgreSQL revocation table to permanently invalidate it.",
)
async def logout_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Explicitly revoke active JWT session token."""
    if auth and auth.credentials:
        revoke_token(db, auth.credentials, current_user.id)
        db.commit()
    return MessageResponse(message="Session successfully revoked.")


@auth_router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user profile",
    description="Returns the profile information of the currently authenticated user.",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Retrieve currently authenticated user profile."""
    return UserResponse.model_validate(current_user)


@auth_router.put(
    "/profile",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update user profile",
    description="Updates the authenticated user's full name in PostgreSQL.",
)
async def update_profile(
    payload: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """Update profile full name for authenticated user."""
    try:
        user = db_service.update_user_profile(db, current_user, payload.full_name)
        db.commit()
        db.refresh(user)
        return UserResponse.model_validate(user)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to update profile: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile.",
        )


@auth_router.post(
    "/change-password",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Change user password",
    description="Verifies the current password and hashes the new password with bcrypt.",
)
async def change_password(
    payload: ChangePasswordRequest,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Verify current password, set new password, and revoke previous token."""
    # 1. Verify current password
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    # 2. Reject if new password equals current password
    if verify_password(payload.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password cannot be the same as the current password.",
        )

    # 3. Check optional confirmation match
    if payload.confirm_password is not None and payload.confirm_password != payload.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password and confirmation password do not match.",
        )

    # 4. Hash new password
    try:
        new_hash = hash_password(payload.new_password)
    except Exception as exc:
        logger.error("Failed to hash new password: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process new password.",
        )

    # 5. Update in database and revoke active token
    try:
        db_service.update_user_password(db, current_user, new_hash)
        if auth and auth.credentials:
            revoke_token(db, auth.credentials, current_user.id)
        db.commit()
        return MessageResponse(message="Password updated successfully.")
    except Exception as exc:
        db.rollback()
        logger.error("Failed to commit password change: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password in database.",
        )


