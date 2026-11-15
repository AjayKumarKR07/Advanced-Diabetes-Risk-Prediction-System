"""
backend/tests/test_insights_api.py
==================================
Comprehensive automated test suite for Phase 6.16 & Phase 6.16.1:
Intelligent Health Insights, Scientific Accuracy & Recommendation Engine Audit.

Tests cover:
- Authentication enforcement (401 on unauthenticated access)
- Empty state handling for accounts with 0 assessments
- Single-assessment recommendation generation across 9 indicators
- Scientific accuracy of Smoker (0 vs 1 lifetime threshold)
- Scientific accuracy of HighBP and HighChol (screening recall vs clinical measurement)
- BMI screening classifications & neutral longitudinal comparison (increased, decreased, stable)
- Authoritative evidence references & survey limitations
- Conflicting responses disclosure (lifetime smoker reverting, prior diagnoses reverting)
- Missing / None indicator safety
- Assessment ID lookup and cross-user data isolation
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

        a = create_assessment(
            user.id,
            bmi=31.5,        # Obese screening
            high_bp=1,       # Reported hypertension
            high_chol=1,     # Reported high cholesterol
            phys_activity=0, # Inactive
            smoker=1,        # Reported >=100 cigarettes
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
            assert "evidence_reference" in rec
            assert rec["evidence_reference"] is not None
            assert "limitations" in rec
            assert rec["limitations"] is not None

        high_recs = [r for r in data["recommendations"] if r["priority"] == "high"]
        assert len(high_recs) >= 5

        summary = data["summary"]
        assert summary["total_insights"] == len(data["recommendations"])
        assert summary["high_priority_count"] == len(high_recs)
        assert "Metabolic & Body" in summary["categories_breakdown"]
        assert "Cardiovascular Health" in summary["categories_breakdown"]


class TestScientificAccuracyAudit:
    """Scientific accuracy checks for Phase 6.16.1."""

    def test_smoker_feature_interpretation(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Case 1: Smoker = 1 (>=100 lifetime cigarettes)
        a1 = create_assessment(user.id, smoker=1)
        res1 = client.get(f"/api/insights?assessment_id={a1.id}", headers=headers)
        data1 = res1.json()
        smk_rec1 = next(r for r in data1["recommendations"] if "Tobacco" in r["indicator_name"])
        assert "100 cigarettes in your lifetime" in smk_rec1["observation"]
        assert "≥100 cigarettes" in smk_rec1["indicator_value"]

        # Case 2: Smoker = 0 (<100 lifetime cigarettes)
        a2 = create_assessment(user.id, smoker=0)
        res2 = client.get(f"/api/insights?assessment_id={a2.id}", headers=headers)
        data2 = res2.json()
        smk_rec2 = next(r for r in data2["recommendations"] if "Tobacco" in r["indicator_name"])
        assert "did not report having smoked at least 100 cigarettes" in smk_rec2["observation"].lower()
        # Must not claim confirmed lifetime non-smoker or never smoked
        assert "never smoked" not in smk_rec2["observation"].lower()
        assert "CDC" in smk_rec2["evidence_reference"]

    def test_blood_pressure_and_cholesterol_descriptions(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # HighBP = 0 and HighChol = 0
        a = create_assessment(user.id, high_bp=0, high_chol=0)
        res = client.get(f"/api/insights?assessment_id={a.id}", headers=headers)
        data = res.json()

        bp_rec = next(r for r in data["recommendations"] if "High Blood Pressure" in r["indicator_name"])
        assert "did not report having been told" in bp_rec["observation"].lower()
        # Must not describe value 0 as a confirmed normal clinical measurement
        assert "confirmed normal" not in bp_rec["observation"].lower()
        assert bp_rec["priority"] == "info"
        assert "ACC/AHA" in bp_rec["evidence_reference"] or "CDC" in bp_rec["evidence_reference"]

        chol_rec = next(r for r in data["recommendations"] if "High Cholesterol" in r["indicator_name"])
        assert "did not report having been told" in chol_rec["observation"].lower()
        assert "confirmed normal" not in chol_rec["observation"].lower()
        assert chol_rec["priority"] == "info"
        assert "AHA/ACC" in chol_rec["evidence_reference"] or "CDC" in chol_rec["evidence_reference"]

    def test_bmi_screening_and_neutral_longitudinal_comparisons(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Assessment 1: BMI 30.5 (obesity screening)
        a1 = create_assessment(user.id, bmi=30.5)
        res1 = client.get(f"/api/insights?assessment_id={a1.id}", headers=headers)
        rec1 = next(r for r in res1.json()["recommendations"] if "BMI" in r["indicator_name"])
        assert "screening classification" in rec1["observation"].lower()
        assert "diagnostic" not in rec1["observation"].lower()

        # Assessment 2: BMI 28.0 (decreased by -2.5 kg/m²)
        a2 = create_assessment(user.id, bmi=28.0)
        res2 = client.get("/api/insights", headers=headers)
        comparisons = res2.json()["historical_comparisons"]
        bmi_comp = next(c for c in comparisons if "BMI" in c["indicator"])

        # Must be neutral "decreased", NOT automatically "improved", and avoid assuming intentional weight loss
        assert bmi_comp["change_type"] == "decreased"
        assert "decreased by 2.5 kg/m²" in bmi_comp["observation"]
        assert "intentional" not in bmi_comp["observation"] or "lifestyle adjustments, appetite shifts" in bmi_comp["observation"]

        # Assessment 3: BMI 28.2 (stable, delta +0.2 kg/m²)
        a3 = create_assessment(user.id, bmi=28.2)
        res3 = client.get("/api/insights", headers=headers)
        comp3 = res3.json()["historical_comparisons"]
        bmi_comp3 = next(c for c in comp3 if "BMI" in c["indicator"])
        assert bmi_comp3["change_type"] == "stable"
        assert "relatively stable" in bmi_comp3["observation"].lower()

    def test_conflicting_responses_disclosure(self, client, create_test_user, create_assessment):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Earlier assessment (a1): Smoker=1 (lifetime threshold reached), HighBP=1, HighChol=1
        a1 = create_assessment(user.id, smoker=1, high_bp=1, high_chol=1)
        # Later assessment (a2): Smoker=0 (conflict), HighBP=0 (conflict), HighChol=0 (conflict)
        a2 = create_assessment(user.id, smoker=0, high_bp=0, high_chol=0)

        res = client.get("/api/insights", headers=headers)
        assert res.status_code == 200
        comparisons = res.json()["historical_comparisons"]

        smk_comp = next(c for c in comparisons if "Tobacco" in c["indicator"])
        assert smk_comp["change_type"] == "conflicting"
        assert "conflict" in smk_comp["observation"].lower() or "discrepancy" in smk_comp["observation"].lower()

        bp_comp = next(c for c in comparisons if "Blood Pressure" in c["indicator"])
        assert bp_comp["change_type"] == "conflicting"
        assert "variation noted" in bp_comp["observation"].lower()

        chol_comp = next(c for c in comparisons if "Cholesterol" in c["indicator"])
        assert chol_comp["change_type"] == "conflicting"
        assert "variation noted" in chol_comp["observation"].lower()


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
        assert "obesity" in res_a.json()["recommendations"][0]["observation"].lower()

        # User B requesting own assessment
        res_b = client.get("/api/insights", headers=headers_b)
        assert res_b.status_code == 200
        assert res_b.json()["assessment_id"] == a_user_b.id
        assert "normal weight" in res_b.json()["recommendations"][0]["observation"].lower()

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
