"""
backend/tests/test_auth_api.py
==============================
Real HTTP integration tests for FastAPI authentication routes:
  - POST /api/auth/register
  - POST /api/auth/login
  - GET  /api/auth/me
  - POST /api/predict (with authenticated user linkage)
"""

from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"qa_test_{uuid.uuid4().hex[:10]}@med-research.org"


class TestAuthEndpointsHTTP:
    def test_register_success_returns_jwt_and_profile(self):
        email = _unique_email()
        res = client.post(
            "/api/auth/register",
            json={
                "full_name": "Dr. Florence Nightingale",
                "email": email,
                "password": "ResearchPassword2026!",
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == email
        assert data["user"]["full_name"] == "Dr. Florence Nightingale"
        assert "hashed_password" not in data["user"]

    def test_register_duplicate_email_returns_409(self):
        email = _unique_email()
        payload = {
            "full_name": "Original User",
            "email": email,
            "password": "Password123!",
        }
        res1 = client.post("/api/auth/register", json=payload)
        assert res1.status_code == 201

        res2 = client.post("/api/auth/register", json=payload)
        assert res2.status_code == 409
        assert "already registered" in res2.json()["detail"].lower()

    def test_register_invalid_email_returns_422(self):
        res = client.post(
            "/api/auth/register",
            json={
                "full_name": "Invalid Email",
                "email": "not-a-valid-email",
                "password": "Password123!",
            },
        )
        assert res.status_code == 422

    def test_register_short_password_returns_422(self):
        res = client.post(
            "/api/auth/register",
            json={
                "full_name": "Short PW User",
                "email": _unique_email(),
                "password": "123",  # < 8 chars
            },
        )
        assert res.status_code == 422

    def test_login_success(self):
        email = _unique_email()
        pw = "MySecretPass2026!"
        client.post(
            "/api/auth/register",
            json={"full_name": "Test Login", "email": email, "password": pw},
        )

        res = client.post(
            "/api/auth/login",
            json={"email": email, "password": pw},
        )
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == email

    def test_login_wrong_password_returns_401(self):
        email = _unique_email()
        client.post(
            "/api/auth/register",
            json={"full_name": "Wrong PW Test", "email": email, "password": "CorrectPassword123!"},
        )

        res = client.post(
            "/api/auth/login",
            json={"email": email, "password": "WrongPassword999!"},
        )
        assert res.status_code == 401
        assert "invalid email or password" in res.json()["detail"].lower()

    def test_login_nonexistent_email_returns_401(self):
        res = client.post(
            "/api/auth/login",
            json={"email": "nobody_exists_here_99999@domain.org", "password": "SomePassword123!"},
        )
        assert res.status_code == 401

    def test_get_me_authenticated(self):
        email = _unique_email()
        reg_res = client.post(
            "/api/auth/register",
            json={"full_name": "Auth Me Test", "email": email, "password": "Password12345!"},
        )
        token = reg_res.json()["access_token"]

        res = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert res.json()["email"] == email
        assert res.json()["full_name"] == "Auth Me Test"

    def test_get_me_unauthenticated_returns_401(self):
        res = client.get("/api/auth/me")
        assert res.status_code == 401

    def test_get_me_invalid_token_returns_401(self):
        res = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_tampered_token_string"},
        )
        assert res.status_code == 401

    def test_predict_with_authenticated_user_links_id(self):
        email = _unique_email()
        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Predict Link Test", "email": email, "password": "Password12345!"},
        )
        token = reg.json()["access_token"]
        user_id = reg.json()["user"]["id"]

        pred_payload = {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 31.0,
            "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
            "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
            "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10,
            "DiffWalk": 1, "Sex": 1, "Age": 9, "Education": 4, "Income": 5,
        }

        res = client.post(
            "/api/predict",
            json=pred_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "assessment_id" in data
        assert "prediction_id" in data

        # Check in DB that the assessment has this user_id
        assess_res = client.get(f"/api/assessments/{data['assessment_id']}")
        assert assess_res.status_code == 200
        assert assess_res.json()["user_id"] == user_id

    def test_update_profile_success(self):
        email = _unique_email()
        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Old Profile Name", "email": email, "password": "Password12345!"},
        )
        token = reg.json()["access_token"]

        res = client.put(
            "/api/auth/profile",
            json={"full_name": "New Valid Profile Name"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert res.json()["full_name"] == "New Valid Profile Name"
        assert res.json()["email"] == email

        # Verify via GET /api/auth/me
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["full_name"] == "New Valid Profile Name"

    def test_update_profile_short_name_returns_422(self):
        email = _unique_email()
        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Testing Short", "email": email, "password": "Password12345!"},
        )
        token = reg.json()["access_token"]

        res = client.put(
            "/api/auth/profile",
            json={"full_name": "A"},  # < 2 chars
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 422

    def test_update_profile_unauthenticated_returns_401(self):
        res = client.put(
            "/api/auth/profile",
            json={"full_name": "Anonymous Attacker"},
        )
        assert res.status_code == 401

    def test_change_password_success(self):
        email = _unique_email()
        old_pw = "OldPassword2026!"
        new_pw = "BrandNewSecret2026!"

        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Password Changer", "email": email, "password": old_pw},
        )
        token = reg.json()["access_token"]

        # Change password
        res = client.post(
            "/api/auth/change-password",
            json={"current_password": old_pw, "new_password": new_pw},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert "successfully" in res.json()["message"].lower()

        # Login with old password must fail (401)
        fail_login = client.post("/api/auth/login", json={"email": email, "password": old_pw})
        assert fail_login.status_code == 401

        # Login with new password must succeed (200)
        success_login = client.post("/api/auth/login", json={"email": email, "password": new_pw})
        assert success_login.status_code == 200
        assert "access_token" in success_login.json()

    def test_change_password_wrong_current_password_returns_400(self):
        email = _unique_email()
        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Wrong PW Tester", "email": email, "password": "CorrectOldPassword1!"},
        )
        token = reg.json()["access_token"]

        res = client.post(
            "/api/auth/change-password",
            json={"current_password": "WrongInputPassword123!", "new_password": "ValidNewPassword99!"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 400
        assert "incorrect" in res.json()["detail"].lower()

    def test_change_password_short_new_password_returns_422(self):
        email = _unique_email()
        reg = client.post(
            "/api/auth/register",
            json={"full_name": "Short New PW", "email": email, "password": "OriginalPassword1!"},
        )
        token = reg.json()["access_token"]

        res = client.post(
            "/api/auth/change-password",
            json={"current_password": "OriginalPassword1!", "new_password": "short"},  # < 8 chars
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 422


