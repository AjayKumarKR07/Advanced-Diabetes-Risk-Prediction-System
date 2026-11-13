"""
backend/tests/test_history_management.py
=========================================
Phase 6.13 Assessment History Management Tests:
  - View saved assessment and associated prediction
  - Deletion of own assessment and cascaded prediction
  - Prevention of unauthorized / cross-user deletion (403 Forbidden)
  - Unauthenticated deletion rejection (401 Unauthorized)
  - Nonexistent assessment deletion (404 Not Found)
  - Stats and count updates after deletion
  - Direct assessment prediction lookup endpoint (/api/assessments/{id}/prediction)
  - Ownership protection against unauthenticated access to registered assessments
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"history_test_{uuid.uuid4().hex[:10]}@med-research.org"


def _register_user(name: str = "History Tester") -> tuple[str, int, str]:
    email = _unique_email()
    pw = "HistoryPass2026!#"
    res = client.post(
        "/api/auth/register",
        json={"full_name": name, "email": email, "password": pw},
    )
    assert res.status_code == 201
    data = res.json()
    return data["access_token"], data["user"]["id"], email


def _sample_features(**overrides) -> dict:
    payload = {
        "HighBP": 1,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 27.2,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 2,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 5,
        "Education": 6,
        "Income": 7,
    }
    payload.update(overrides)
    return payload


class TestAssessmentHistoryManagement:
    """Test suite for Phase 6.13 History management capabilities."""

    def test_view_saved_assessment_and_prediction(self):
        token, user_id, _ = _register_user("View Tester")
        pred_res = client.post(
            "/api/predict",
            json=_sample_features(BMI=26.4, HighBP=1, Age=4),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert pred_res.status_code == 200
        data = pred_res.json()
        assess_id = data["assessment_id"]
        pred_id = data["prediction_id"]

        # View assessment record
        get_assess = client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_assess.status_code == 200
        assess_json = get_assess.json()
        assert assess_json["id"] == assess_id
        assert assess_json["user_id"] == user_id
        assert assess_json["BMI"] == 26.4
        assert assess_json["HighBP"] == 1
        assert assess_json["Age"] == 4

        # View prediction via assessment endpoint
        get_pred = client.get(
            f"/api/assessments/{assess_id}/prediction",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_pred.status_code == 200
        pred_json = get_pred.json()
        assert pred_json["id"] == pred_id
        assert pred_json["assessment_id"] == assess_id
        assert "predicted_class" in pred_json
        assert "confidence" in pred_json

    def test_delete_own_assessment_success_cascades_prediction(self):
        token, user_id, _ = _register_user("Delete Tester")
        pred_res = client.post(
            "/api/predict",
            json=_sample_features(BMI=31.0),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert pred_res.status_code == 200
        data = pred_res.json()
        assess_id = data["assessment_id"]
        pred_id = data["prediction_id"]

        # Perform deletion
        del_res = client.delete(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert del_res.status_code == 200
        del_json = del_res.json()
        assert del_json["assessment_id"] == assess_id
        assert "deleted" in del_json["message"].lower()

        # Confirm assessment is no longer found
        assert client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code == 404

        # Confirm associated prediction is cascaded and deleted
        assert client.get(
            f"/api/predictions/{pred_id}",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code == 404

    def test_delete_another_user_assessment_forbidden(self):
        token_a, user_a_id, _ = _register_user("User A Owner")
        token_b, user_b_id, _ = _register_user("User B Attacker")

        # User A creates assessment
        pred_res = client.post(
            "/api/predict",
            json=_sample_features(BMI=24.5),
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assess_id = pred_res.json()["assessment_id"]

        # User B attempts to delete User A's assessment
        del_res = client.delete(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert del_res.status_code == 403
        assert "forbidden" in del_res.json()["detail"].lower()

        # Confirm assessment was NOT deleted
        check_res = client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert check_res.status_code == 200

    def test_delete_unauthenticated_returns_401(self):
        token, _, _ = _register_user("Auth Req Tester")
        pred_res = client.post(
            "/api/predict",
            json=_sample_features(BMI=22.1),
            headers={"Authorization": f"Bearer {token}"},
        )
        assess_id = pred_res.json()["assessment_id"]

        # Unauthenticated delete attempt
        del_res = client.delete(f"/api/assessments/{assess_id}")
        assert del_res.status_code == 401

    def test_delete_nonexistent_assessment_returns_404(self):
        token, _, _ = _register_user("404 Tester")
        del_res = client.delete(
            "/api/assessments/99999999",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert del_res.status_code == 404
        assert "not found" in del_res.json()["detail"].lower()

    def test_stats_and_history_update_after_deletion(self):
        token, user_id, _ = _register_user("Stats Update Tester")

        # Submit 2 assessments
        res1 = client.post(
            "/api/predict",
            json=_sample_features(BMI=23.0),
            headers={"Authorization": f"Bearer {token}"},
        )
        res2 = client.post(
            "/api/predict",
            json=_sample_features(BMI=33.0),
            headers={"Authorization": f"Bearer {token}"},
        )
        a1_id = res1.json()["assessment_id"]
        a2_id = res2.json()["assessment_id"]

        # Verify initial stats = 2
        stats_before = client.get("/api/stats", headers={"Authorization": f"Bearer {token}"}).json()
        assert stats_before["total_assessments"] == 2
        assert stats_before["total_predictions"] == 2

        # Delete assessment 1
        del_res = client.delete(f"/api/assessments/{a1_id}", headers={"Authorization": f"Bearer {token}"})
        assert del_res.status_code == 200

        # Verify stats updated = 1
        stats_after = client.get("/api/stats", headers={"Authorization": f"Bearer {token}"}).json()
        assert stats_after["total_assessments"] == 1
        assert stats_after["total_predictions"] == 1

        # Verify assessments list returns only 1 item
        list_res = client.get("/api/assessments", headers={"Authorization": f"Bearer {token}"}).json()
        assert list_res["total"] == 1
        assert list_res["items"][0]["id"] == a2_id

    def test_cross_user_prediction_lookup_forbidden(self):
        token_a, _, _ = _register_user("Pred Owner A")
        token_b, _, _ = _register_user("Pred Attacker B")

        res_a = client.post(
            "/api/predict",
            json=_sample_features(BMI=25.0),
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assess_id = res_a.json()["assessment_id"]

        # User B attempts to access User A's prediction via assessment endpoint
        res_b = client.get(
            f"/api/assessments/{assess_id}/prediction",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_b.status_code == 403

    def test_cross_user_access_strictly_forbidden(self):
        token_a, _, _ = _register_user("Private Assessee A")
        token_b, _, _ = _register_user("Private Assessee B")

        res_a = client.post(
            "/api/predict",
            json=_sample_features(BMI=28.0),
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assess_id = res_a.json()["assessment_id"]
        pred_id = res_a.json()["prediction_id"]

        # User B tries to view User A's assessment -> 403
        foreign_assess = client.get(
            f"/api/assessments/{assess_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert foreign_assess.status_code == 403

        # User B tries to view User A's prediction -> 403
        foreign_pred = client.get(
            f"/api/predictions/{pred_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert foreign_pred.status_code == 403

        # User B tries to view prediction via assessment endpoint -> 403
        foreign_pred_by_assess = client.get(
            f"/api/assessments/{assess_id}/prediction",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert foreign_pred_by_assess.status_code == 403
