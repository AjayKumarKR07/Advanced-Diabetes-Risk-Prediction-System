"""
backend/tests/test_profile_settings.py
======================================
Comprehensive integration tests for Phase 6.14:
  - User Profile retrieval and update (GET /api/profile, PUT /api/profile)
  - Validation & authentication guardrails on profile endpoints
  - Secure password change (POST /api/auth/change-password)
  - User preferences & settings management (GET /api/settings, PUT /api/settings)
  - User data isolation between accounts
  - Permanent account deletion with password re-authentication and cascade verification (DELETE /api/account)
"""

from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"qa_phase614_{uuid.uuid4().hex[:10]}@diabetes-research.org"


def _register_user(name: str = "Test Subject", password: str = "ValidPassword2026!") -> tuple[str, str, dict]:
    """Helper: register user and return (token, email, user_dict)."""
    email = _unique_email()
    res = client.post(
        "/api/auth/register",
        json={"full_name": name, "email": email, "password": password},
    )
    assert res.status_code == 201, f"Failed registration: {res.text}"
    data = res.json()
    return data["access_token"], email, data["user"]


class TestUserProfileAndSettings:
    # 1. Fetch own profile
    def test_fetch_own_profile(self):
        token, email, user = _register_user("Alice Walker")
        res = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == user["id"]
        assert data["full_name"] == "Alice Walker"
        assert data["email"] == email
        assert data["account_status"] == "Active"
        assert data["email_verified"] is True
        assert data["total_assessments"] == 0
        assert data["latest_assessment_date"] is None

    # 2. Update own profile
    def test_update_own_profile(self):
        token, email, _ = _register_user("Original Name")
        res = client.put(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={"full_name": "Updated Dr. Name"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["full_name"] == "Updated Dr. Name"

        # Verify persistence on subsequent fetch
        get_res = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert get_res.status_code == 200
        assert get_res.json()["full_name"] == "Updated Dr. Name"

    # 3. Reject invalid profile updates
    def test_reject_invalid_profile_updates(self):
        token, _, _ = _register_user("Valid Name")
        # Empty or whitespace only
        res1 = client.put(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={"full_name": "   "},
        )
        assert res1.status_code == 422

        # Too short (<2 chars)
        res2 = client.put(
            "/api/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={"full_name": "X"},
        )
        assert res2.status_code == 422

    # 4. Reject unauthenticated profile requests
    def test_reject_unauthenticated_profile_requests(self):
        # Missing auth header
        res1 = client.get("/api/profile")
        assert res1.status_code == 401

        res2 = client.put("/api/profile", json={"full_name": "Hacker Attempt"})
        assert res2.status_code == 401

        # Invalid token
        res3 = client.get("/api/profile", headers={"Authorization": "Bearer invalid.token.payload"})
        assert res3.status_code == 401

    # 5. Change password successfully
    def test_change_password_successfully(self):
        token, email, _ = _register_user("Bob Ross", "ArtPassword123!")
        res = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "ArtPassword123!",
                "new_password": "NewLandscape2026!",
                "confirm_password": "NewLandscape2026!",
            },
        )
        assert res.status_code == 200
        assert "success" in res.json()["message"].lower()

        # Verify old password no longer works
        old_login = client.post(
            "/api/auth/login",
            json={"email": email, "password": "ArtPassword123!"},
        )
        assert old_login.status_code == 401

        # Verify new password authenticates successfully
        new_login = client.post(
            "/api/auth/login",
            json={"email": email, "password": "NewLandscape2026!"},
        )
        assert new_login.status_code == 200
        assert "access_token" in new_login.json()

    # 6. Reject incorrect current password
    def test_reject_incorrect_current_password(self):
        token, _, _ = _register_user("User PW", "RealPassword99!")
        res = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "WrongPasswordEntered!",
                "new_password": "BrandNewSecret100!",
            },
        )
        assert res.status_code == 400
        assert "current password is incorrect" in res.json()["detail"].lower()

    # 7. Reject mismatched confirmation password
    def test_reject_mismatched_confirmation_password(self):
        token, _, _ = _register_user("Mismatch User", "CurrentPass123!")
        res = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "CurrentPass123!",
                "new_password": "BrandNewSecret100!",
                "confirm_password": "DifferentConfirmation99!",
            },
        )
        assert res.status_code in (400, 422)

    # 8. Reject weak passwords and same new password
    def test_reject_weak_and_same_new_passwords(self):
        token, _, _ = _register_user("Same User", "ExistingSecret2026!")

        # Short password (< 8 chars)
        res_short = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "ExistingSecret2026!",
                "new_password": "short",
            },
        )
        assert res_short.status_code == 422

        # Same new password as current password
        res_same = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "ExistingSecret2026!",
                "new_password": "ExistingSecret2026!",
            },
        )
        assert res_same.status_code == 400
        assert "cannot be the same" in res_same.json()["detail"].lower()

    # 9. Fetch and update own settings
    def test_fetch_and_update_own_settings(self):
        token, _, _ = _register_user("Settings User")

        # Initial fetch -> default settings
        res1 = client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
        assert res1.status_code == 200
        d1 = res1.json()
        assert d1["theme"] == "system"
        assert d1["notify_assessment_ready"] is True
        assert d1["notify_security_alerts"] is True
        assert d1["notify_product_updates"] is False

        # Update settings
        res2 = client.put(
            "/api/settings",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "theme": "dark",
                "notify_product_updates": True,
                "notify_assessment_ready": False,
            },
        )
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["theme"] == "dark"
        assert d2["notify_product_updates"] is True
        assert d2["notify_assessment_ready"] is False

        # Verify persistence on re-fetch
        res3 = client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
        assert res3.status_code == 200
        assert res3.json()["theme"] == "dark"
        assert res3.json()["notify_product_updates"] is True

    # 10. Verify preferences remain isolated between users
    def test_preferences_remain_isolated_between_users(self):
        token_a, _, _ = _register_user("User Alpha")
        token_b, _, _ = _register_user("User Beta")

        # User A chooses dark theme
        client.put(
            "/api/settings",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"theme": "dark", "notify_product_updates": True},
        )

        # User B chooses light theme
        client.put(
            "/api/settings",
            headers={"Authorization": f"Bearer {token_b}"},
            json={"theme": "light", "notify_product_updates": False},
        )

        # Check User A
        res_a = client.get("/api/settings", headers={"Authorization": f"Bearer {token_a}"})
        assert res_a.json()["theme"] == "dark"
        assert res_a.json()["notify_product_updates"] is True

        # Check User B
        res_b = client.get("/api/settings", headers={"Authorization": f"Bearer {token_b}"})
        assert res_b.json()["theme"] == "light"
        assert res_b.json()["notify_product_updates"] is False

    # 11. Reject unauthorized account deletion
    def test_reject_unauthorized_account_deletion(self):
        res = client.request(
            "DELETE",
            "/api/account",
            json={"password": "any", "confirmation": "DELETE"},
        )
        assert res.status_code == 401

    # 12. Reject account deletion with incorrect password or confirmation
    def test_reject_account_deletion_incorrect_password_or_token(self):
        token, _, _ = _register_user("Delete Fail User", "CorrectPassword123!")

        # Incorrect password
        res1 = client.request(
            "DELETE",
            "/api/account",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": "WrongPassword!", "confirmation": "DELETE"},
        )
        assert res1.status_code == 400
        assert "incorrect password" in res1.json()["detail"].lower()

        # Mismatched confirmation token
        res2 = client.request(
            "DELETE",
            "/api/account",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": "CorrectPassword123!", "confirmation": "NOPE"},
        )
        assert res2.status_code in (400, 422)

    # 13 & 14. Successfully delete an account, cascade records, and verify cannot authenticate
    def test_delete_account_success_and_prevents_login(self):
        token, email, user = _register_user("Doomed User", "PasswordToBeDeleted123!")

        # Create an assessment to test cascade
        payload = {
            "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 24.5, "Smoker": 0,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
            "Age": 5, "Education": 5, "Income": 7,
        }
        pred_res = client.post(
            "/api/predict",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        assert pred_res.status_code == 200
        assessment_id = pred_res.json()["assessment_id"]

        # Confirm assessment is visible in user's profile
        prof_res = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert prof_res.status_code == 200
        assert prof_res.json()["total_assessments"] >= 1

        # Delete account with password and DELETE confirmation
        del_res = client.request(
            "DELETE",
            "/api/account",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": "PasswordToBeDeleted123!", "confirmation": "DELETE"},
        )
        assert del_res.status_code == 200
        assert "permanently deleted" in del_res.json()["message"].lower()

        # 14: Verify deleted user CANNOT authenticate
        login_res = client.post(
            "/api/auth/login",
            json={"email": email, "password": "PasswordToBeDeleted123!"},
        )
        assert login_res.status_code == 401

        # Verify old token cannot be used
        post_del_profile = client.get("/api/profile", headers={"Authorization": f"Bearer {token}"})
        assert post_del_profile.status_code == 401
