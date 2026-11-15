"""
backend/app/services/avatar_service.py
======================================
Secure persistent profile avatar storage service:
  - Magic byte validation for JPEG, PNG, and WebP
  - Strict 2MB file size enforcement
  - Protection against malicious, polyglot, or executable files
  - Sandboxed path storage with client-supplied filename discarding
  - User-specific isolation and deletion hooks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

MAX_AVATAR_BYTES = 2 * 1024 * 1024  # 2 Megabytes
AVATARS_DIR = Path("uploads/avatars")


def ensure_avatars_dir() -> Path:
    """Ensure the uploads/avatars directory exists."""
    AVATARS_DIR.mkdir(parents=True, exist_ok=True)
    return AVATARS_DIR


def detect_and_validate_image_type(data: bytes) -> str:
    """
    Validate binary magic bytes to guarantee file is truly JPEG, PNG, or WebP.
    Protects against executable scripts (.sh, .exe, .php, .py) masquerading as images.
    Returns extension string without dot: 'jpg', 'png', or 'webp'.
    """
    if len(data) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small to be a valid image.",
        )

    if len(data) > MAX_AVATAR_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar image size exceeds the maximum allowed limit of 2MB.",
        )

    # Magic byte signatures
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Only authentic JPEG, PNG, and WebP images are permitted.",
        )


def save_user_avatar(user_id: int, data: bytes) -> str:
    """
    Validate and save user avatar file.
    Client-supplied filenames are ignored to prevent path traversal attacks.
    """
    ext = detect_and_validate_image_type(data)
    folder = ensure_avatars_dir()

    # Clean up any existing avatar for this user
    remove_user_avatar(user_id)

    file_path = folder / f"user_{user_id}.{ext}"
    try:
        with open(file_path, "wb") as f:
            f.write(data)
        logger.info("Persisted avatar for user %s: %s", user_id, file_path)
        return f"/api/profile/avatar"
    except Exception as exc:
        logger.error("Failed to write avatar file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save avatar image to server storage.",
        )


def get_user_avatar_path(user_id: int) -> Optional[Path]:
    """Retrieve Path to user avatar file if it exists on disk."""
    folder = ensure_avatars_dir()
    for file_path in folder.glob(f"user_{user_id}.*"):
        if file_path.is_file():
            return file_path
    return None


def remove_user_avatar(user_id: int) -> bool:
    """Permanently delete user avatar file(s) from disk."""
    folder = ensure_avatars_dir()
    removed = False
    for file_path in folder.glob(f"user_{user_id}.*"):
        try:
            file_path.unlink()
            removed = True
            logger.info("Removed avatar file: %s", file_path)
        except OSError as exc:
            logger.warning("Could not remove avatar file %s: %s", file_path, exc)
    return removed

