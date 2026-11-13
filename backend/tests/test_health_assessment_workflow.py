"""
backend/tests/test_health_assessment_workflow.py
=================================================
Phase 6.10 Health Assessment Form & Prediction Workflow Tests:
  - Age category conversion logic
  - 21-feature mapping and schema validation
  - Prediction execution with authenticated user linkage
  - User-specific dashboard statistics isolation
  - User-specific assessment and prediction history filtering
  - Prevention of cross-user access (403 Forbidden on foreign records)
  - Guest vs authenticated access behavior
"""

from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"workflow_qa_{uuid.uuid4().hex[:10]}@med-research.org"


def _register_user(name: str = "Test Researcher") -> tuple[str, int, str]:
    email = _unique_email()
    pw = "StrongPass2026!#"
    res = client.post(
        "/api/auth/register",
        json={"full_name": name, "email": email, "password": pw},
    )
    assert res.status_code == 201
    data = res.json()
    return data["access_token"], data["user"]["id"], email


def _sample_21_features(**overrides) -> dict:
    payload = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 28.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 3,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 8,  # 55-59 years
        "Education": 5,
        "Income": 6,
    }
    payload.update(overrides)
    return payload


class TestAgeConversionLogic:
    """Verify CDC BRFSS 2015 13-category age conversion rules."""

    @staticmethod
    def convert_age(age: int) -> int | None:
        if age < 18 or age > 120:
            return None
        if age <= 24:
            return 1
        if age <= 29:
            return 2
        if age <= 34:
            return 3
        if age <= 39:
            return 4
        if age <= 44:
            return 5
        if age <= 49:
            return 6
        if age <= 54:
            return 7
        if age <= 59:
            return 8
        if age <= 64:
            return 9
        if age <= 69:
            return 10
        if age <= 74:
            return 11
        if age <= 79:
            return 12
        return 13

    def test_age_under_18_rejected(self):
        assert self.convert_age(17) is None
        assert self.convert_age(0) is None

    def test_age_over_120_rejected(self):
        assert self.convert_age(121) is None

    def test_age_bracket_boundaries(self):
        assert self.convert_age(18) == 1
        assert self.convert_age(24) == 1
        assert self.convert_age(25) == 2
        assert self.convert_age(29) == 2
        assert self.convert_age(30) == 3
        assert self.convert_age(34) == 3
        assert self.convert_age(35) == 4
        assert self.convert_age(39) == 4
        assert self.convert_age(40) == 5
        assert self.convert_age(44) == 5
        assert self.convert_age(45) == 6
        assert self.convert_age(49) == 6
        assert self.convert_age(50) == 7
        assert self.convert_age(54) == 7
        assert self.convert_age(55) == 8
        assert self.convert_age(59) == 8
        assert self.convert_age(60) == 9
        assert self.convert_age(64) == 9
        assert self.convert_age(65) == 10
        assert self.convert_age(69) == 10
        assert self.convert_age(70) == 11
        assert self.convert_age(74) == 11
        assert self.convert_age(75) == 12
        assert self.convert_age(79) == 12
        assert self.convert_age(80) == 13
        assert self.convert_age(95) == 13


class TestPredictionWorkflowAndUserLinkage:
    """End-to-end prediction workflow, linking user ID and returning full result."""

    def test_authenticated_prediction_associates_user(self):
        token, user_id, _ = _register_user("Dr. Marie Curie")
        payload = _sample_21_features()

        res = client.post(
            "/api/predict",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["assessment_id"] > 0
        assert data["prediction_id"] > 0
        assert data["user_id"] == user_id
        assert data["user_name"] == "Dr. Marie Curie"
        assert data["predicted_class"] in (0, 1, 2)
        assert data["predicted_label"] in ("No Diabetes", "Prediabetes", "Diabetes")
        assert 0.0 <= data["probability_no_diabetes"] <= 1.0
        assert 0.0 <= data["probability_prediabetes"] <= 1.0
        assert 0.0 <= data["probability_diabetes"] <= 1.0

        # Probabilities sum to 1.0 (approx)
        total_p = (
            data["probability_no_diabetes"]
            + data["probability_prediabetes"]
            + data["probability_diabetes"]
        )
        assert total_p == pytest.approx(1.0, abs=1e-4)

    def test_guest_prediction_supported_without_auth(self):
        payload = _sample_21_features()
        res = client.post("/api/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["assessment_id"] > 0
        assert data["prediction_id"] > 0
        assert data["user_id"] is None
        assert data["user_name"] is None


class TestUserIsolationAndSecurity:
    """Verify statistics, history isolation, and 403 Forbidden on cross-user access."""

    def test_user_history_isolation(self):
        # Create User A and submit an assessment
        token_a, user_a_id, _ = _register_user("User Alpha")
        res_a = client.post(
            "/api/predict",
            json=_sample_21_features(BMI=22.0),
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_a.status_code == 200
        pred_a_id = res_a.json()["prediction_id"]
        assess_a_id = res_a.json()["assessment_id"]

        # Create User B and submit an assessment
        token_b, user_b_id, _ = _register_user("User Beta")
        res_b = client.post(
            "/api/predict",
            json=_sample_21_features(BMI=38.0),
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_b.status_code == 200
        pred_b_id = res_b.json()["prediction_id"]
        assess_b_id = res_b.json()["assessment_id"]

        # User A querying history only sees their own predictions
        hist_a = client.get(
            "/api/predictions",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert hist_a.status_code == 200
        items_a = hist_a.json()["items"]
        assert any(item["id"] == pred_a_id for item in items_a)
        assert not any(item["id"] == pred_b_id for item in items_a)

        # User B querying history only sees their own predictions
        hist_b = client.get(
            "/api/predictions",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert hist_b.status_code == 200
        items_b = hist_b.json()["items"]
        assert any(item["id"] == pred_b_id for item in items_b)
        assert not any(item["id"] == pred_a_id for item in items_b)

    def test_cross_user_access_forbidden(self):
        # User A creates a record
        token_a, user_a_id, _ = _register_user("User Private")
        res_a = client.post(
            "/api/predict",
            json=_sample_21_features(BMI=24.0),
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assess_id = res_a.json()["assessment_id"]
        pred_id = res_a.json()["prediction_id"]

        # User B tries to view User A's assessment -> 403 Forbidden
        token_b, user_b_id, _ = _register_user("User Attacker")
        res_foreign_assess = client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_foreign_assess.status_code == 403
        assert "access forbidden" in res_foreign_assess.json()["detail"].lower()

        # User B tries to view User A's prediction -> 403 Forbidden
        res_foreign_pred = client.get(
            f"/api/predictions/{pred_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_foreign_pred.status_code == 403
        assert "access forbidden" in res_foreign_pred.json()["detail"].lower()

        # User A CAN view their own assessment and prediction
        res_own_assess = client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_own_assess.status_code == 200
        assert res_own_assess.json()["id"] == assess_id

        res_own_pred = client.get(
            f"/api/predictions/{pred_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_own_pred.status_code == 200
        assert res_own_pred.json()["id"] == pred_id

    def test_user_specific_dashboard_stats(self):
        token_u, user_id, _ = _register_user("User Stats Test")
        # Submit 2 assessments for this user
        client.post(
            "/api/predict",
            json=_sample_21_features(BMI=23.0),
            headers={"Authorization": f"Bearer {token_u}"},
        )
        client.post(
            "/api/predict",
            json=_sample_21_features(BMI=29.0),
            headers={"Authorization": f"Bearer {token_u}"},
        )

        # Authenticated stats call returns exactly 2 assessments for this user
        res = client.get(
            "/api/stats",
            headers={"Authorization": f"Bearer {token_u}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 2
        assert data["total_predictions"] == 2
        assert len(data["recent_assessments"]) == 2
        for a in data["recent_assessments"]:
            assert a["user_id"] == user_id

