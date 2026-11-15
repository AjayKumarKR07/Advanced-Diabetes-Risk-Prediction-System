"""
backend/app/core/security.py
============================
Authentication & security utilities:
  - Cryptographic password hashing & verification via bcrypt
  - JWT token encoding & decoding via PyJWT (HS256)
  - FastAPI dependencies for authenticated routes (get_current_user, get_optional_user)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
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

def hash_token(raw_token: str) -> str:
    """Generate SHA-256 fingerprint of a raw JWT string."""
    return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()


def create_access_token(
    user_id: int,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generate a signed JWT access token with a unique jti for an authenticated user."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    jti = str(uuid.uuid4())
    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": jti,
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


def revoke_token(db: Session, token: str, user_id: int) -> bool:
    """
    Record an access token in the PostgreSQL revoked_tokens table.
    Subsequent requests with this token will be rejected immediately (HTTP 401).
    """
    try:
        from backend.app.models.revoked_token import RevokedToken

        payload = decode_access_token(token)
        token_h = hash_token(token)

        existing = db.scalar(
            select(RevokedToken.id).where(RevokedToken.token_hash == token_h)
        )
        if existing:
            return True

        if payload and payload.get("jti"):
            jti = payload["jti"]
        else:
            jti = token_h[:32]

        exp_ts = payload.get("exp") if payload else None
        if exp_ts:
            expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
        else:
            expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        revoked_record = RevokedToken(
            jti=jti,
            token_hash=token_h,
            user_id=user_id,
            expires_at=expires_at,
        )
        db.add(revoked_record)
        db.flush()
        return True
    except Exception as exc:
        logger.error("Failed to persist revoked token: %s", exc)
        return False


def is_token_revoked(db: Session, token: str) -> bool:
    """Verify whether a token has been explicitly revoked in PostgreSQL."""
    try:
        from backend.app.models.revoked_token import RevokedToken

        token_h = hash_token(token)
        revoked_id = db.scalar(
            select(RevokedToken.id).where(RevokedToken.token_hash == token_h)
        )
        return revoked_id is not None
    except Exception as exc:
        logger.error("Error checking token revocation status: %s", exc)
        return False


# ---------------------------------------------------------------------------
# FastAPI authentication dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that enforces authentication and verifies token revocation.
    Raises HTTP 401 if token is missing, invalid, expired, revoked, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or session has expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if auth is None or not auth.credentials:
        raise credentials_exception

    # Check server-side revocation list in PostgreSQL
    if is_token_revoked(db, auth.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
    Dependency for routes where authentication is optional.
    Returns None if token is missing, invalid, expired, or revoked.
    """
    if auth is None or not auth.credentials:
        return None

    if is_token_revoked(db, auth.credentials):
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

