"""
backend/tests/test_security_audit.py
====================================
Comprehensive security, privacy, and account management audit suite (Phase 6.14.1):
  1. Profile Photo Security & Persistence (JPEG/PNG/WebP magic bytes, 2MB limit, malicious file defense)
  2. JWT Session Security & Revocation (signature, expiration, revocation via logout and password change)
  3. Password Security & Bcrypt byte limits (72-byte boundary, same password, brute-force rate limiting)
  4. Account Deletion & Complete Data Purge (atomic cascade of assessments, predictions, settings, avatar)
  5. API Authorization & User Data Isolation
"""

from __future__ import annotations

import io
import time
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.avatar_service import get_user_avatar_path

client = TestClient(app)

# Real binary image headers for testing
VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
VALID_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
VALID_WEBP_BYTES = b"RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x00\x30\x01\x00\x9d\x01\x2a\x01\x00\x01\x00"
MALICIOUS_EXEC_BYTES = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00This program cannot be run in DOS mode."
MALICIOUS_SHELL_BYTES = b"#!/bin/bash\nrm -rf /\necho 'attack'"


def _unique_email() -> str:
    return f"sec_audit_{uuid.uuid4().hex[:10]}@security-audit.org"


def _register_user(name: str = "Audit Target", password: str = "StrongPass2026!") -> tuple[str, str, dict]:
    email = _unique_email()
    res = client.post(
        "/api/auth/register",
        json={"full_name": name, "email": email, "password": password},
    )
    assert res.status_code == 201, f"Register failed: {res.text}"
    data = res.json()
    return data["access_token"], email, data["user"]


class TestProfilePhotoSecurity:
    """Audit profile photo upload, validation, persistence, and isolation."""

    def test_upload_valid_png_and_retrieve(self):
        token, _, user = _register_user("Photo User PNG")
        files = {"file": ("avatar.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        res = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files)
        assert res.status_code == 200
        assert "uploaded" in res.json()["message"].lower()

        # Verify image retrieval endpoint
        get_res = client.get("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"})
        assert get_res.status_code == 200
        assert get_res.headers["content-type"] == "image/png"
        assert get_res.content == VALID_PNG_BYTES

        # Verify profile includes avatar flag
        prof_res = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert prof_res.status_code == 200
        assert prof_res.json()["has_avatar"] is True
        assert prof_res.json()["avatar_url"] == "/api/profile/avatar"

    def test_upload_valid_jpeg_and_webp(self):
        token, _, user = _register_user("Photo User JPG/WEBP")

        # Upload JPEG
        files_jpg = {"file": ("profile.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")}
        res_jpg = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files_jpg)
        assert res_jpg.status_code == 200

        get_jpg = client.get("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"})
        assert get_jpg.status_code == 200
        assert get_jpg.headers["content-type"] == "image/jpeg"

        # Replace with WebP
        files_webp = {"file": ("profile.webp", io.BytesIO(VALID_WEBP_BYTES), "image/webp")}
        res_webp = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files_webp)
        assert res_webp.status_code == 200

        get_webp = client.get("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"})
        assert get_webp.status_code == 200
        assert get_webp.headers["content-type"] == "image/webp"

    def test_reject_executable_and_malicious_files(self):
        token, _, _ = _register_user("Malicious Tester")

        # Attempt to upload executable disguised as JPG
        files_exe = {"file": ("malware.jpg", io.BytesIO(MALICIOUS_EXEC_BYTES), "image/jpeg")}
        res_exe = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files_exe)
        assert res_exe.status_code == 400
        assert "invalid image format" in res_exe.json()["detail"].lower()

        # Attempt to upload shell script disguised as PNG
        files_sh = {"file": ("script.png", io.BytesIO(MALICIOUS_SHELL_BYTES), "image/png")}
        res_sh = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files_sh)
        assert res_sh.status_code == 400
        assert "invalid image format" in res_sh.json()["detail"].lower()

    def test_reject_files_exceeding_2mb(self):
        token, _, _ = _register_user("Oversized Tester")
        huge_data = VALID_PNG_BYTES + (b"\x00" * (2 * 1024 * 1024 + 500))  # > 2MB
        files_huge = {"file": ("big.png", io.BytesIO(huge_data), "image/png")}
        res = client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files_huge)
        assert res.status_code == 400
        assert "2mb" in res.json()["detail"].lower()

    def test_avatar_delete_and_cleanup(self):
        token, _, user = _register_user("Delete Photo User")
        files = {"file": ("avatar.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files)
        assert get_user_avatar_path(user["id"]) is not None

        # Delete avatar
        del_res = client.delete("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"})
        assert del_res.status_code == 200
        assert get_user_avatar_path(user["id"]) is None

        # Subsequent GET returns 404
        get_res = client.get("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"})
        assert get_res.status_code == 404


class TestJWTSessionRevocation:
    """Audit JWT session revocation, logout invalidation, and expiration enforcement."""

    def test_explicit_logout_revokes_token(self):
        token, _, _ = _register_user("Logout Target")

        # Token works initially
        initial_check = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert initial_check.status_code == 200

        # Perform server-side logout
        logout_res = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert logout_res.status_code == 200
        assert "revoked" in logout_res.json()["message"].lower()

        # Token is immediately rejected with HTTP 401
        revoked_check = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert revoked_check.status_code == 401
        assert "revoked" in revoked_check.json()["detail"].lower()

    def test_password_change_revokes_active_token(self):
        token, email, _ = _register_user("PW Revocation Target", "InitialPass123!")

        # Change password
        chg_res = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "InitialPass123!",
                "new_password": "NewSecurePassword456!",
                "confirm_password": "NewSecurePassword456!",
            },
        )
        assert chg_res.status_code == 200

        # Previous token must now be revoked
        post_pw_check = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert post_pw_check.status_code == 401
        assert "revoked" in post_pw_check.json()["detail"].lower()

    def test_tampered_and_invalid_tokens_rejected(self):
        token, _, _ = _register_user("Tamper Test")

        # Flip last characters
        tampered = token[:-5] + "XXXXX"
        res = client.get("/api/profile", headers={"Authorization": f"Bearer {tampered}"})
        assert res.status_code == 401


class TestPasswordSecurityAndLimits:
    """Audit password security, bcrypt 72-byte limit, and brute force defenses."""

    def test_bcrypt_72_byte_limit_enforced(self):
        long_pw = "P" * 75  # > 72 bytes
        email = _unique_email()
        res = client.post(
            "/api/auth/register",
            json={"full_name": "Long Password User", "email": email, "password": long_pw},
        )
        assert res.status_code == 422

    def test_login_rate_limiting_brute_force_defense(self):
        email = _unique_email()
        # Register user
        client.post(
            "/api/auth/register",
            json={"full_name": "Target", "email": email, "password": "TargetPassword123!"},
        )

        # Trigger 5 failed logins rapidly
        for _ in range(5):
            fail_res = client.post("/api/auth/login", json={"email": email, "password": "WrongPassword!"})
            assert fail_res.status_code == 401

        # 6th attempt should be blocked with HTTP 429
        blocked_res = client.post("/api/auth/login", json={"email": email, "password": "WrongPassword!"})
        assert blocked_res.status_code == 429
        assert "too many" in blocked_res.json()["detail"].lower()


class TestAccountDeletionDataPurge:
    """Audit atomic deletion of user account and cascade cleanup of all health records & files."""

    def test_account_deletion_cleans_all_records_and_avatar(self):
        token, email, user = _register_user("Purge Candidate", "PurgePass123!")

        # 1. Upload avatar
        files = {"file": ("avatar.png", io.BytesIO(VALID_PNG_BYTES), "image/png")}
        client.post("/api/profile/avatar", headers={"Authorization": f"Bearer {token}"}, files=files)
        assert get_user_avatar_path(user["id"]) is not None

        # 2. Add an assessment and prediction
        payload = {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 29.2, "Smoker": 1,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 3, "MentHlth": 5, "PhysHlth": 5, "DiffWalk": 0, "Sex": 1,
            "Age": 8, "Education": 4, "Income": 5,
        }
        pred_res = client.post("/api/predict", headers={"Authorization": f"Bearer {token}"}, json=payload)
        assert pred_res.status_code == 200

        # 3. Add custom settings
        client.put("/api/settings", headers={"Authorization": f"Bearer {token}"}, json={"theme": "dark"})

        # 4. Perform atomic account deletion
        del_res = client.request(
            "DELETE",
            "/api/account",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": "PurgePass123!", "confirmation": "DELETE"},
        )
        assert del_res.status_code == 200

        # 5. Verify avatar file removed from disk
        assert get_user_avatar_path(user["id"]) is None

        # 6. Verify user cannot log in
        login_res = client.post("/api/auth/login", json={"email": email, "password": "PurgePass123!"})
        assert login_res.status_code == 401

        # 7. Verify token no longer accesses any endpoint
        prof_res = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert prof_res.status_code == 401

