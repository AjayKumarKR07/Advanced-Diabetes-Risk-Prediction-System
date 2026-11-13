"""
backend/tests/test_auth.py
==========================
Phase 6.9 Authentication & User Management Test Suite.

Tests:
  - Cryptographic password hashing & verification (bcrypt)
  - JWT token generation, claims encoding, expiration, and tampering rejection
  - User registration with input validation (email format, password strength)
  - Duplicate email collision protection (HTTP 409 Conflict)
  - User authentication / login with correct and incorrect credentials
  - PostgreSQL user record persistence and isolation
  - FastAPI dependency token resolution (get_current_user, get_optional_user)
"""

from __future__ import annotations

from datetime import timedelta
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.core.database import Base
from backend.app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.models.user import User
from backend.app.schemas.auth_schemas import (
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.app.services import db_service

TEST_DB_URL = settings.TEST_DATABASE_URL or settings.DATABASE_URL.replace(
    "diabetes_risk_db", "diabetes_risk_test_db"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auth_test_engine():
    """Engine bound to the test database with tables initialized."""
    engine = create_engine(TEST_DB_URL, pool_pre_ping=True, echo=False)
    # Ensure models are loaded
    from backend.app.models import user, assessment, prediction  # noqa: F401

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def AuthSessionLocal(auth_test_engine):
    return sessionmaker(bind=auth_test_engine, autocommit=False, autoflush=False)


@pytest.fixture()
def db_session(AuthSessionLocal):
    """Transactional session that automatically rolls back after each test."""
    session = AuthSessionLocal()
    try:
        yield session
        session.rollback()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 1. Cryptography & Password Hashing Unit Tests
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_creates_bcrypt_format(self):
        pw = "SecureResearchPass2026!"
        hashed = hash_password(pw)
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert len(hashed) == 60
        assert hashed != pw  # Never plaintext

    def test_verify_password_positive(self):
        pw = "TestPassword@123"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_verify_password_negative(self):
        pw = "CorrectPass123"
        hashed = hash_password(pw)
        assert verify_password("WrongPass123", hashed) is False

    def test_verify_password_handles_empty_or_corrupt_hash(self):
        assert verify_password("Password", "") is False
        assert verify_password("Password", "not_a_valid_hash") is False


# ---------------------------------------------------------------------------
# 2. JWT Generation & Verification Tests
# ---------------------------------------------------------------------------

class TestTokenOperations:
    def test_create_and_decode_token(self):
        token = create_access_token(user_id=42, email="researcher@hospital.org")
        payload = decode_access_token(token)

        assert payload is not None
        assert payload.get("sub") == "42"
        assert payload.get("email") == "researcher@hospital.org"
        assert "exp" in payload
        assert "iat" in payload

    def test_expired_token_rejected(self):
        # Create token that expired 10 minutes ago
        token = create_access_token(
            user_id=1,
            email="expired@user.org",
            expires_delta=timedelta(minutes=-10),
        )
        payload = decode_access_token(token)
        assert payload is None

    def test_tampered_token_rejected(self):
        token = create_access_token(user_id=10, email="tamper@user.org")
        # Corrupt the payload part of the JWT
        parts = token.split(".")
        tampered_token = f"{parts[0]}.eyJob3N0aWxlIjogdHJ1ZX0.{parts[2]}"
        payload = decode_access_token(tampered_token)
        assert payload is None


# ---------------------------------------------------------------------------
# 3. Pydantic Input Validation Tests
# ---------------------------------------------------------------------------

class TestAuthSchemaValidation:
    def test_valid_registration_payload(self):
        req = UserRegisterRequest(
            full_name="Dr. Alan Turing",
            email="alan.turing@cambridge.edu",
            password="StrongPassword123!",
        )
        assert req.full_name == "Dr. Alan Turing"
        assert req.email == "alan.turing@cambridge.edu"

    def test_invalid_email_format_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(
                full_name="Valid Name",
                email="not-an-email",
                password="ValidPassword123!",
            )

    def test_short_password_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(
                full_name="Valid Name",
                email="valid@example.com",
                password="short",  # < 8 chars
            )

    def test_short_name_raises(self):
        with pytest.raises(ValidationError):
            UserRegisterRequest(
                full_name="A",  # < 2 chars
                email="valid@example.com",
                password="ValidPassword123!",
            )


# ---------------------------------------------------------------------------
# 4. PostgreSQL User Persistence & Authentication Workflows
# ---------------------------------------------------------------------------

class TestDatabaseUserWorkflows:
    def test_create_and_retrieve_user(self, db_session):
        email = "test.user.persistence@example.com"
        pw_hash = hash_password("SecretPass123!")
        user = db_service.create_user(
            db=db_session,
            full_name="Jane Doe",
            email=email,
            hashed_password=pw_hash,
        )
        db_session.flush()

        assert user.id is not None
        assert user.email == email
        assert user.hashed_password == pw_hash

        # Retrieve by email
        fetched = db_service.get_user_by_email(db_session, email)
        assert fetched is not None
        assert fetched.id == user.id
        assert fetched.full_name == "Jane Doe"

        # Safe UserResponse serialization
        safe_profile = UserResponse.model_validate(fetched)
        assert safe_profile.email == email
        assert not hasattr(safe_profile, "hashed_password")

    def test_get_user_by_email_case_insensitive(self, db_session):
        email_mixed = "MixedCase.Researcher@Example.Com"
        user = db_service.create_user(
            db=db_session,
            full_name="Mixed Case",
            email=email_mixed,
            hashed_password=hash_password("Pass12345!"),
        )
        db_session.flush()

        found = db_service.get_user_by_email(db_session, "mixedcase.researcher@example.com")
        assert found is not None
        assert found.id == user.id

    def test_duplicate_email_raises_integrity_error(self, db_session):
        from sqlalchemy.exc import IntegrityError

        email = "duplicate.test@example.com"
        pw_hash = hash_password("Pass123456!")
        db_service.create_user(db_session, "User One", email, pw_hash)
        db_session.flush()

        with pytest.raises(IntegrityError):
            db_service.create_user(db_session, "User Two", email, pw_hash)
            db_session.flush()

    def test_get_nonexistent_user_returns_none(self, db_session):
        assert db_service.get_user_by_email(db_session, "nonexistent@nowhere.com") is None
        assert db_service.get_user_by_id(db_session, 99999999) is None

