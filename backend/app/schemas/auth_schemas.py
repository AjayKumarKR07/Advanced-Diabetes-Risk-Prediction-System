"""
backend/app/schemas/auth_schemas.py
===================================
Pydantic schemas for user registration, login, and authentication tokens.
Ensures strong validation on email and password inputs.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
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
    confirm_password: Optional[str] = Field(None, max_length=128, description="Optional confirm new password for dual check")

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long.")
        return v

    @field_validator("confirm_password")
    @classmethod
    def validate_confirmation_match(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None and "new_password" in info.data:
            if v != info.data["new_password"]:
                raise ValueError("New password and confirmation password do not match.")
        return v


class UserProfileResponse(BaseModel):
    """Enriched user profile with longitudinal account activity statistics."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str
    created_at: datetime
    total_assessments: int = 0
    latest_assessment_date: Optional[datetime] = None
    account_status: str = "Active"
    email_verified: bool = True


class UserSettingsResponse(BaseModel):
    """User preferences and application settings."""
    model_config = ConfigDict(from_attributes=True)

    theme: str = "system"
    notify_assessment_ready: bool = True
    notify_security_alerts: bool = True
    notify_product_updates: bool = False
    updated_at: Optional[datetime] = None


class UserSettingsUpdateRequest(BaseModel):
    """Payload for updating user preferences."""
    theme: Optional[str] = Field(None, description="Preferred UI theme ('system', 'light', 'dark')")
    notify_assessment_ready: Optional[bool] = Field(None, description="Email preference for assessment completion")
    notify_security_alerts: Optional[bool] = Field(None, description="Security alert notifications")
    notify_product_updates: Optional[bool] = Field(None, description="Product and methodology update announcements")

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip().lower()
            if clean not in {"system", "light", "dark"}:
                raise ValueError("Theme must be one of: 'system', 'light', 'dark'.")
            return clean
        return v


class AccountDeleteRequest(BaseModel):
    """Payload required to permanently delete an authenticated user account."""
    password: str = Field(..., min_length=1, max_length=128, description="Current account password for re-authentication")
    confirmation: str = Field(..., description="Confirmation token string (must equal 'DELETE')")

    @field_validator("confirmation")
    @classmethod
    def validate_confirmation_token(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean != "DELETE":
            raise ValueError("Confirmation token must be exactly 'DELETE'.")
        return clean


class MessageResponse(BaseModel):
    """Generic status message response."""
    message: str


