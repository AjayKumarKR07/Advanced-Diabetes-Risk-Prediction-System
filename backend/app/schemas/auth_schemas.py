"""
backend/app/schemas/auth_schemas.py
===================================
Pydantic schemas for user registration, login, and authentication tokens.
Ensures strong validation on email and password inputs.
"""

from __future__ import annotations

import re
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator


EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$")


class UserRegisterRequest(BaseModel):
    """User registration payload."""
    full_name: str = Field(..., min_length=2, max_length=120, description="Full name of user")
    email: str = Field(..., max_length=255, description="Valid email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email address format.")
        return clean

    @field_validator("full_name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Full name must be at least 2 characters.")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class UserLoginRequest(BaseModel):
    """User login payload."""
    email: str = Field(..., max_length=255, description="Registered email address")
    password: str = Field(..., min_length=1, max_length=128, description="Account password")

    @field_validator("email")
    @classmethod
    def sanitize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    """Safe user profile response — never exposes password hash."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str
    created_at: datetime


class TokenResponse(BaseModel):
    """JWT bearer token and authenticated user profile."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserProfileUpdateRequest(BaseModel):
    """User profile update payload."""
    full_name: str = Field(..., min_length=2, max_length=120, description="Updated full name")

    @field_validator("full_name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Full name must be at least 2 characters.")
        return cleaned


class ChangePasswordRequest(BaseModel):
    """Password change payload."""
    current_password: str = Field(..., min_length=1, max_length=128, description="Current account password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (min 8 characters)")

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long.")
        return v


class MessageResponse(BaseModel):
    """Generic status message response."""
    message: str


