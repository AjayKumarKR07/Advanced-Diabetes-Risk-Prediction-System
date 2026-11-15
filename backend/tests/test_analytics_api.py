"""
backend/tests/test_analytics_api.py
===================================
Automated test suite for Phase 6.15 Personal Health Analytics API.
Tests cover:
- Authentication enforcement (401 on missing or invalid tokens)
- Empty state analytics for users with 0 assessments
- Single and multiple assessment analytics calculations (BMI trends, GenHlth trends)
- Date range filtering (start_date, end_date)
- Prediction class distribution and percentages
- Health indicator longitudinal comparison
- Assessment activity timeline with pagination
- Cross-user data isolation
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
    def _create(email=None, name="Test Analytics User"):
        if email is None:
            email = f"analytics_{uuid.uuid4().hex[:8]}@example.com"
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
def create_assessment_with_pred(db_session):
    def _create(user_id, bmi=24.5, gen_hlth=2, pred_class=0, pred_label="No Diabetes", confidence=0.75):
        assessment = Assessment(
            user_id=user_id,
            HighBP=0,
            HighChol=0,
            CholCheck=1,
            Smoker=0,
            Stroke=0,
            HeartDiseaseorAttack=0,
            PhysActivity=1,
            Fruits=1,
            Veggies=1,
            HvyAlcoholConsump=0,
            AnyHealthcare=1,
            NoDocbcCost=0,
            DiffWalk=0,
            Sex=1,
            BMI=bmi,
            MentHlth=1,
            PhysHlth=0,
            GenHlth=gen_hlth,
            Age=5,
            Education=6,
            Income=8,
        )
        db_session.add(assessment)
        db_session.flush()

        pred = Prediction(
            assessment_id=assessment.id,
            predicted_class=pred_class,
            predicted_label=pred_label,
            risk_tier=f"Class {pred_class}",
            confidence=confidence,
            probability_no_diabetes=0.75 if pred_class == 0 else 0.15,
            probability_prediabetes=0.15 if pred_class == 1 else 0.10,
            probability_diabetes=0.75 if pred_class == 2 else 0.15,
            model_version="LogisticRegression-Phase5",
        )
        db_session.add(pred)
        db_session.commit()
        db_session.refresh(assessment)
        return assessment, pred
    return _create


class TestAnalyticsAuthentication:
    def test_unauthenticated_requests_return_401(self, client):
        endpoints = [
            "/api/analytics/overview",
            "/api/analytics/prediction-distribution",
            "/api/analytics/bmi-trends",
            "/api/analytics/health-trends",
            "/api/analytics/indicators",
            "/api/analytics/timeline",
        ]
        for ep in endpoints:
            res = client.get(ep)
            assert res.status_code == 401, f"Expected 401 for unauthenticated request to {ep}"


class TestAnalyticsCalculations:
    def test_empty_analytics_new_user(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # Overview
        res = client.get("/api/analytics/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 0
        assert data["assessments_this_month"] == 0
        assert data["latest_assessment_id"] is None
        assert data["latest_predicted_class"] is None
        assert data["latest_bmi"] is None
        assert data["bmi_summary"]["count"] == 0
        assert data["bmi_summary"]["latest"] is None

        # Distribution
        res_dist = client.get("/api/analytics/prediction-distribution", headers=headers)
        assert res_dist.status_code == 200
        dist_data = res_dist.json()
        assert dist_data["total"] == 0
        assert dist_data["counts"]["0"] == 0

        # Trends
        res_bmi = client.get("/api/analytics/bmi-trends", headers=headers)
        assert res_bmi.status_code == 200
        assert len(res_bmi.json()["points"]) == 0

        # Timeline
        res_timeline = client.get("/api/analytics/timeline", headers=headers)
        assert res_timeline.status_code == 200
        assert res_timeline.json()["total"] == 0
        assert len(res_timeline.json()["items"]) == 0

    def test_single_assessment_analytics(self, client, create_test_user, create_assessment_with_pred):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}
        create_assessment_with_pred(user.id, bmi=22.4, gen_hlth=2, pred_class=0, pred_label="No Diabetes")

        res = client.get("/api/analytics/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 1
        assert data["assessments_this_month"] == 1
        assert data["latest_bmi"] == 22.4
        assert "Normal range" in data["latest_bmi_category"]
        assert data["latest_predicted_class"] == 0
        assert data["latest_predicted_label"] == "No Diabetes"
        assert data["bmi_summary"]["latest"] == 22.4
        assert data["bmi_summary"]["previous"] is None
        assert data["bmi_summary"]["change"] is None  # Needs >= 2 points

    def test_multiple_assessments_trends_and_changes(self, client, create_test_user, create_assessment_with_pred):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # First assessment (earlier)
        create_assessment_with_pred(user.id, bmi=27.5, gen_hlth=3, pred_class=1, pred_label="Prediabetes")
        # Second assessment (newer)
        create_assessment_with_pred(user.id, bmi=25.0, gen_hlth=2, pred_class=0, pred_label="No Diabetes")

        # Overview check
        res = client.get("/api/analytics/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assessments"] == 2
        assert data["latest_bmi"] == 25.0
        assert data["bmi_summary"]["latest"] == 25.0
        assert data["bmi_summary"]["previous"] == 27.5
        assert data["bmi_summary"]["change"] == -2.5

        # GenHlth summary check
        assert data["gen_hlth_summary"]["latest"] == 2
        assert data["gen_hlth_summary"]["previous"] == 3
        assert data["gen_hlth_summary"]["change"] == -1

        # Distribution check
        res_dist = client.get("/api/analytics/prediction-distribution", headers=headers)
        assert res_dist.status_code == 200
        dist = res_dist.json()
        assert dist["total"] == 2
        assert dist["counts"]["0"] == 1
        assert dist["counts"]["1"] == 1
        assert dist["counts"]["2"] == 0
        assert dist["percentages"]["0"] == 50.0
        assert dist["percentages"]["1"] == 50.0

        # BMI Trends endpoint
        res_bmi = client.get("/api/analytics/bmi-trends", headers=headers)
        assert res_bmi.status_code == 200
        bmi_data = res_bmi.json()
        assert len(bmi_data["points"]) == 2
        assert bmi_data["points"][0]["bmi"] == 27.5  # chronological order (oldest first)
        assert bmi_data["points"][1]["bmi"] == 25.0
        assert bmi_data["summary"]["change"] == -2.5

        # Indicator comparison endpoint
        res_ind = client.get("/api/analytics/indicators", headers=headers)
        assert res_ind.status_code == 200
        ind_data = res_ind.json()
        assert ind_data["total_records"] == 2
        assert ind_data["records"][0]["bmi"] == 25.0  # newest first in comparison table

        # Timeline endpoint
        res_time = client.get("/api/analytics/timeline?page=1&page_size=10", headers=headers)
        assert res_time.status_code == 200
        time_data = res_time.json()
        assert time_data["total"] == 2
        assert len(time_data["items"]) == 2
        assert time_data["items"][0]["predicted_class"] == 0

    def test_date_range_filtering_presets_and_custom(self, client, create_test_user, create_assessment_with_pred):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}
        create_assessment_with_pred(user.id, bmi=26.0, gen_hlth=2, pred_class=0, pred_label="No Diabetes")

        # Future date range where no records exist
        res_empty = client.get(
            "/api/analytics/overview?start_date=2099-01-01&end_date=2099-12-31",
            headers=headers,
        )
        assert res_empty.status_code == 200
        data_empty = res_empty.json()
        assert data_empty["total_assessments"] == 0
        assert data_empty["latest_assessment_id"] is None
        assert data_empty["bmi_summary"]["count"] == 0

        # Wide range including all
        res_all = client.get(
            "/api/analytics/overview?start_date=2020-01-01&end_date=2030-12-31",
            headers=headers,
        )
        assert res_all.status_code == 200
        assert res_all.json()["total_assessments"] == 1


class TestAnalyticsDataIsolation:
    def test_user_analytics_strictly_isolated(self, client, create_test_user, create_assessment_with_pred):
        user_a, token_a = create_test_user(name="User Alpha")
        user_b, token_b = create_test_user(name="User Beta")

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # User A has 2 assessments
        create_assessment_with_pred(user_a.id, bmi=31.2, gen_hlth=4, pred_class=2, pred_label="Diabetes")
        create_assessment_with_pred(user_a.id, bmi=29.8, gen_hlth=3, pred_class=1, pred_label="Prediabetes")

        # User B has 1 assessment
        create_assessment_with_pred(user_b.id, bmi=21.0, gen_hlth=1, pred_class=0, pred_label="No Diabetes")

        # Check User A
        res_a = client.get("/api/analytics/overview", headers=headers_a)
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["total_assessments"] == 2
        assert data_a["latest_bmi"] == 29.8
        assert data_a["class_distribution"]["2"] == 1
        assert data_a["class_distribution"]["0"] == 0

        # Check User B
        res_b = client.get("/api/analytics/overview", headers=headers_b)
        assert res_b.status_code == 200
        data_b = res_b.json()
        assert data_b["total_assessments"] == 1
        assert data_b["latest_bmi"] == 21.0
        assert data_b["class_distribution"]["0"] == 1
        assert data_b["class_distribution"]["2"] == 0

    def test_cross_user_isolation_all_sub_endpoints(self, client, create_test_user, create_assessment_with_pred):
        user_a, token_a = create_test_user(name="User Isolated A")
        user_b, token_b = create_test_user(name="User Isolated B")

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        create_assessment_with_pred(user_a.id, bmi=33.3, gen_hlth=5, pred_class=2, pred_label="Diabetes")
        create_assessment_with_pred(user_b.id, bmi=20.2, gen_hlth=1, pred_class=0, pred_label="No Diabetes")

        # Distribution isolation
        dist_a = client.get("/api/analytics/prediction-distribution", headers=headers_a).json()
        dist_b = client.get("/api/analytics/prediction-distribution", headers=headers_b).json()
        assert dist_a["counts"]["2"] == 1
        assert dist_a["counts"]["0"] == 0
        assert dist_b["counts"]["0"] == 1
        assert dist_b["counts"]["2"] == 0

        # BMI Trends isolation
        bmi_a = client.get("/api/analytics/bmi-trends", headers=headers_a).json()
        bmi_b = client.get("/api/analytics/bmi-trends", headers=headers_b).json()
        assert len(bmi_a["points"]) == 1
        assert bmi_a["points"][0]["bmi"] == 33.3
        assert len(bmi_b["points"]) == 1
        assert bmi_b["points"][0]["bmi"] == 20.2

        # Health Trends isolation
        hlth_a = client.get("/api/analytics/health-trends", headers=headers_a).json()
        hlth_b = client.get("/api/analytics/health-trends", headers=headers_b).json()
        assert hlth_a["points"][0]["gen_hlth"] == 5
        assert hlth_b["points"][0]["gen_hlth"] == 1

        # Indicators isolation
        ind_a = client.get("/api/analytics/indicators", headers=headers_a).json()
        ind_b = client.get("/api/analytics/indicators", headers=headers_b).json()
        assert ind_a["records"][0]["diff_walk"] == 0
        assert ind_a["records"][0]["bmi"] == 33.3
        assert ind_b["records"][0]["bmi"] == 20.2

        # Timeline isolation
        time_a = client.get("/api/analytics/timeline", headers=headers_a).json()
        time_b = client.get("/api/analytics/timeline", headers=headers_b).json()
        assert time_a["items"][0]["predicted_class"] == 2
        assert time_b["items"][0]["predicted_class"] == 0


class TestAnalyticsSecurityAndValidation:
    def test_invalid_date_format_returns_400(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        res_bad_start = client.get("/api/analytics/overview?start_date=not-a-date", headers=headers)
        assert res_bad_start.status_code == 400
        assert "Invalid start_date format" in res_bad_start.json()["detail"]

        res_bad_end = client.get("/api/analytics/overview?end_date=bad-date", headers=headers)
        assert res_bad_end.status_code == 400
        assert "Invalid end_date format" in res_bad_end.json()["detail"]

    def test_inverted_date_range_returns_400(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        res = client.get(
            "/api/analytics/overview?start_date=2026-12-01&end_date=2026-11-01",
            headers=headers,
        )
        assert res.status_code == 400
        assert "start_date (2026-12-01) cannot be after end_date" in res.json()["detail"]

    def test_sql_injection_attempt_in_date_filter_safely_rejected(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        res = client.get(
            "/api/analytics/overview?start_date=2026-01-01' OR 1=1 --",
            headers=headers,
        )
        assert res.status_code == 400
        assert "Invalid start_date format" in res.json()["detail"]

    def test_pagination_boundaries_and_rejections(self, client, create_test_user):
        user, token = create_test_user()
        headers = {"Authorization": f"Bearer {token}"}

        # page_size > 100 rejected with 422
        res_large = client.get("/api/analytics/timeline?page=1&page_size=500", headers=headers)
        assert res_large.status_code == 422

        # page < 1 rejected with 422
        res_zero = client.get("/api/analytics/timeline?page=0&page_size=20", headers=headers)
        assert res_zero.status_code == 422
