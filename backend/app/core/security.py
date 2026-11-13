"""
backend/app/core/security.py
============================
Authentication & security utilities:
  - Cryptographic password hashing & verification via bcrypt
  - JWT token encoding & decoding via PyJWT (HS256)
  - FastAPI dependencies for authenticated routes (get_current_user, get_optional_user)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.user import User

logger = logging.getLogger(__name__)

# HTTPBearer scheme: extracts "Bearer <token>" from Authorization header
security_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Password hashing & verification
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Generate a secure salted bcrypt hash for a plaintext password."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Password verification error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# JWT creation & verification
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generate a signed JWT access token for an authenticated user."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }

    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT access token. Returns payload dict or None."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.info("JWT access token expired.")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid JWT access token: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FastAPI authentication dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that enforces authentication.
    Raises HTTP 401 if token is missing, invalid, expired, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or session has expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if auth is None or not auth.credentials:
        raise credentials_exception

    payload = decode_access_token(auth.credentials)
    if payload is None:
        raise credentials_exception

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise credentials_exception

    user = db.get(User, user_id)
    if user is None:
        raise credentials_exception

    return user


async def get_optional_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Dependency for routes where authentication is optional (e.g. assessments
    can be submitted by guests OR logged-in users).
    """
    if auth is None or not auth.credentials:
        return None

    payload = decode_access_token(auth.credentials)
    if payload is None:
        return None

    user_id_str = payload.get("sub")
    if user_id_str is None:
        return None

    try:
        user_id = int(user_id_str)
        return db.get(User, user_id)
    except ValueError:
        return None

