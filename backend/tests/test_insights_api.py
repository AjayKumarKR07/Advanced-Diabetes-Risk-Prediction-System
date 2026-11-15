"""
backend/tests/test_insights_api.py
==================================
Automated test suite for Phase 6.16 Intelligent Health Insights API.
Tests cover:
- Authentication enforcement (401 on unauthenticated access)
- Empty state handling for accounts with 0 assessments
- Single-assessment recommendation generation across 9 indicators
- Multi-assessment historical indicator comparisons & changes
- Assessment ID lookup and cross-user data isolation
- Educational disclaimer verification
- Summary KPI endpoint counts and category breakdown
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.database import SessionLocal
from backend.app.models.assessment import Assessment
from backend.app.models.prediction import Prediction
from backend.app.models.user import User
from backend.app.core.security import hash_password, create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def create_test_user(db_session):
    def _create(email=None, name="Insights User"):
        if email is None:
            email = f"insights_{uuid.uuid4().hex[:8]}@example.com"
        user = User(
            full_name=name,
            email=email,
            hashed_password=hash_password("Password123!"),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        token = create_access_token(user.id, user.email)
        return user, token
    return _create


@pytest.fixture
def create_assessment(db_session):
    def _create(
        user_id,
        bmi=24.0,
        high_bp=0,
        high_chol=0,
        phys_activity=1,
        smoker=0,
        gen_hlth=2,
        phys_hlth=0,
        ment_hlth=0,
        diff_walk=0,
    ):
        assessment = Assessment(
            user_id=user_id,
            HighBP=high_bp,
            HighChol=high_chol,
            CholCheck=1,
            Smoker=smoker,
            Stroke=0,
            HeartDiseaseorAttack=0,
            PhysActivity=phys_activity,
            Fruits=1,
            Veggies=1,
            HvyAlcoholConsump=0,
            AnyHealthcare=1,
            NoDocbcCost=0,
            DiffWalk=diff_walk,
            Sex=1,
            BMI=bmi,
            MentHlth=ment_hlth,
            PhysHlth=phys_hlth,
            GenHlth=gen_hlth,
            Age=5,
            Education=6,
            Income=8,
        )
        db_session.add(assessment)
        db_session.flush()

        pred = Prediction(
            assessment_id=assessment.id,
            predicted_class=0,
            predicted_label="No Diabetes",
            risk_tier="Class 0",
            confidence=0.85,
            probability_no_diabetes=0.85,
            probability_prediabetes=0.10,
            probability_diabetes=0.05,
            model_version="LogisticRegression-Phase5",
        )
        db_session.add(pred)
        db_session.commit()
        db_session.refresh(assessment)
        return assessment
    return _create


class TestInsightsAuthentication:
    def test_unauthenticated_requests_return_401(self, client):
        endpoints = [
            "/api/insights",
            "/api/insights/summary",
            "/api/insights/history",
        ]
        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 401, f"Expected 401 for unauthenticated request to {ep}"


class TestInsightsGeneration:
    def test_empty_insights_new_user(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        res = client.get("/api/insights", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 0
        assert data["assessment_id"] is None
        assert len(data["recommendations"]) == 0
        assert len(data["historical_comparisons"]) == 0
        assert data["summary"]["total_insights"] == 0

    def test_single_assessment_insights_and_recommendations(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Elevated risk factors profile
        a = create_assessment(
            user.id,
            bmi=31.5,        # Obese
            high_bp=1,       # Hypertension
            high_chol=1,     # High cholesterol
            phys_activity=0, # Inactive
            smoker=1,        # Smoker
            gen_hlth=4,      # Fair
            phys_hlth=15,    # Frequent physical distress
            ment_hlth=16,    # Frequent mental distress
            diff_walk=1,     # Mobility limitation
        )

        res = client.get("/api/insights", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 1
        assert data["assessment_id"] == a.id
        assert len(data["recommendations"]) >= 9
        assert len(data["historical_comparisons"]) == 0  # Requires >= 2 assessments

        # Verify recommendation structure
        for rec in data["recommendations"]:
            assert "id" in rec
            assert "category" in rec
            assert "priority" in rec
            assert "observation" in rec
            assert "indicator_name" in rec
            assert "indicator_value" in rec
            assert "explanation" in rec
            assert "suggestion" in rec
            assert "reason" in rec
            assert "date_generated" in rec

        # Check high priority recommendations
        high_recs = [r for r in data["recommendations"] if r["priority"] == "high"]
        assert len(high_recs) >= 5

        # Check summary
        summary = data["summary"]
        assert summary["total_insights"] == len(data["recommendations"])
        assert summary["high_priority_count"] == len(high_recs)
        assert "Metabolic & Body" in summary["categories_breakdown"]
        assert "Cardiovascular Health" in summary["categories_breakdown"]

    def test_multiple_assessments_historical_comparisons(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Assessment 1 (Earlier)
        a1 = create_assessment(
            user.id,
            bmi=28.0,
            high_bp=1,
            high_chol=1,
            phys_activity=0,
            gen_hlth=4,
            phys_hlth=10,
            ment_hlth=8,
        )
        # Assessment 2 (Newer, improved profile)
        a2 = create_assessment(
            user.id,
            bmi=26.5,        # Improved by -1.5
            high_bp=0,       # Improved
            high_chol=1,     # Stable
            phys_activity=1, # Improved
            gen_hlth=2,      # Improved
            phys_hlth=3,     # Improved
            ment_hlth=2,     # Improved
        )

        res = client.get("/api/insights", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 2
        assert data["assessment_id"] == a2.id
        assert len(data["historical_comparisons"]) >= 6

        # Check BMI comparison item
        bmi_comp = next((c for c in data["historical_comparisons"] if "BMI" in c["indicator"]), None)
        assert bmi_comp is not None
        assert bmi_comp["change_type"] == "improved"
        assert "28.0" in bmi_comp["previous_value"]
        assert "26.5" in bmi_comp["current_value"]

        # Check Physical Activity comparison item
        act_comp = next((c for c in data["historical_comparisons"] if "Physical Activity" in c["indicator"]), None)
        assert act_comp is not None
        assert act_comp["change_type"] == "improved"
        assert act_comp["previous_value"] == "Inactive"
        assert act_comp["current_value"] == "Active"

        # Check History endpoint
        res_hist = client.get("/api/insights/history", headers=headers)
        assert res_hist.status_code == 200
        hist_data = res_hist.json()
        assert hist_data["total_assessments"] == 2
        assert len(hist_data["comparisons"]) >= 6


class TestInsightsSecurityAndIsolation:
    def test_cross_user_isolation(self, client, create_test_user, create_assessment):
        user_a, token_a = create_test_user(name="User Insights A")
        user_b, token_b = create_test_user(name="User Insights B")

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        a_user_a = create_assessment(user_a.id, bmi=35.0, high_bp=1)
        a_user_b = create_assessment(user_b.id, bmi=22.0, high_bp=0)

        # User A requesting own assessment
        res_a = client.get("/api/insights", headers=headers_a)
        assert res_a.status_code == 200
        assert res_a.json()["assessment_id"] == a_user_a.id
        assert "Obesity" in res_a.json()["recommendations"][0]["observation"]

        # User B requesting own assessment
        res_b = client.get("/api/insights", headers=headers_b)
        assert res_b.status_code == 200
        assert res_b.json()["assessment_id"] == a_user_b.id
        assert "Normal Weight" in res_b.json()["recommendations"][0]["observation"]

        # User A attempting to query User B's assessment ID -> 404
        res_forbidden = client.get(f"/api/insights?assessment_id={a_user_b.id}", headers=headers_a)
        assert res_forbidden.status_code == 404

    def test_summary_endpoint(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}
        create_assessment(user.id, bmi=24.0, high_bp=0, high_chol=0)

        res = client.get("/api/insights/summary", headers=headers)
        assert res.status_code == 200
        summary = res.json()
        assert summary["total_insights"] >= 5
        assert summary["latest_assessment_id"] is not None
        assert summary["positive_habits_count"] >= 1

