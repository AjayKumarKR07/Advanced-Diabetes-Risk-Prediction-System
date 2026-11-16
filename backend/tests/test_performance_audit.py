"""
backend/tests/test_performance_audit.py
=======================================
Phase 6.17 Production Readiness, Security & Performance Benchmark Test Suite.

Verifies:
  - Liveness probe (/health) and Readiness probe (/ready)
  - Request Correlation ID middleware (X-Request-ID propagation)
  - Production Security Headers (nosniff, DENY, XSS-Protection, Referrer-Policy)
  - Production settings validation logic
  - Latency benchmarks for /predict, analytics, pagination, and insights
  - Concurrent authenticated load resilience
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import Settings, DEV_FALLBACK_SECRET_KEY
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
def test_user_with_token(db_session):
    email = f"perf_{uuid.uuid4().hex[:8]}@example.com"
    user = User(
        full_name="Performance Audit User",
        email=email,
        hashed_password=hash_password("PerfAudit123!"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    token = create_access_token(user.id, user.email)
    return user, token


@pytest.fixture
def populate_test_assessments(db_session, test_user_with_token):
    user, _ = test_user_with_token
    # Populate 15 representative historical assessments
    for i in range(15):
        a = Assessment(
            user_id=user.id,
            HighBP=1 if i % 2 == 0 else 0,
            HighChol=1 if i % 3 == 0 else 0,
            CholCheck=1,
            Smoker=1 if i % 4 == 0 else 0,
            Stroke=0,
            HeartDiseaseorAttack=0,
            PhysActivity=1 if i % 2 == 1 else 0,
            Fruits=1,
            Veggies=1,
            HvyAlcoholConsump=0,
            AnyHealthcare=1,
            NoDocbcCost=0,
            DiffWalk=0,
            Sex=1,
            BMI=24.0 + (i * 0.4),
            MentHlth=i % 5,
            PhysHlth=i % 4,
            GenHlth=2 + (i % 3),
            Age=5,
            Education=6,
            Income=8,
        )
        db_session.add(a)
        db_session.flush()

        p = Prediction(
            assessment_id=a.id,
            predicted_class=0 if i < 10 else 1,
            predicted_label="No Diabetes" if i < 10 else "Prediabetes",
            risk_tier="Class 0" if i < 10 else "Class 1",
            confidence=0.82,
            probability_no_diabetes=0.82,
            probability_prediabetes=0.12,
            probability_diabetes=0.06,
            model_version="LogisticRegression-Phase5",
        )
        db_session.add(p)

    db_session.commit()
    return user


class TestProductionEndpointsAndSecurity:
    def test_liveness_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "version" in data

    def test_readiness_endpoint(self, client):
        res = client.get("/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["database_connected"] is True

    def test_correlation_id_middleware(self, client):
        # 1. Custom Request ID provided
        custom_id = "req-custom-audit-12345"
        res = client.get("/health", headers={"X-Request-ID": custom_id})
        assert res.status_code == 200
        assert res.headers.get("X-Request-ID") == custom_id

        # 2. Auto-generated Request ID
        res_auto = client.get("/health")
        assert res_auto.status_code == 200
        assert "X-Request-ID" in res_auto.headers
        assert len(res_auto.headers["X-Request-ID"]) > 10

    def test_production_security_headers(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert res.headers.get("X-XSS-Protection") == "1; mode=block"
        assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_production_settings_validation(self):
        # Test 1: Production with default dev key fails
        prod_settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY=DEV_FALLBACK_SECRET_KEY,
            CORS_ORIGINS=["*"],
            DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/db",
        )
        errors = prod_settings.validate_production_settings()
        assert len(errors) >= 3
        assert any("SECRET_KEY" in e for e in errors)
        assert any("Wildcard" in e for e in errors)
        assert any("postgres:postgres" in e for e in errors)

        # Test 2: Production with secure settings passes
        secure_settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="c" * 64,
            CORS_ORIGINS=["https://app.diabetesrisk.ai"],
            DATABASE_URL="postgresql+psycopg://valid_user:strong_pass@prod-db.internal:5432/db",
        )
        secure_errors = secure_settings.validate_production_settings()
        assert len(secure_errors) == 0


class TestBackendPerformanceBenchmarks:
    """Quantitative performance benchmarks measuring mean, p50, and p95 latency."""

    def test_predict_endpoint_latency(self, client, test_user_with_token):
        _, token = test_user_with_token
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 28.5,
            "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1,
            "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
            "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5,
            "DiffWalk": 0, "Sex": 1, "Age": 8, "Education": 5, "Income": 6,
        }

        latencies = []
        # Warmup
        client.post("/api/predict", json=payload, headers=headers)

        for _ in range(30):
            t0 = time.perf_counter()
            res = client.post("/api/predict", json=payload, headers=headers)
            t1 = time.perf_counter()
            assert res.status_code == 200
            latencies.append((t1 - t0) * 1000)

        latencies.sort()
        mean_lat = sum(latencies) / len(latencies)
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\n[BENCHMARK] /api/predict: Mean={mean_lat:.2f}ms, P50={p50:.2f}ms, P95={p95:.2f}ms")
        assert mean_lat < 100.0, f"Predict endpoint latency ({mean_lat:.2f}ms) exceeded threshold"

    def test_analytics_and_insights_latency(self, client, test_user_with_token, populate_test_assessments):
        _, token = test_user_with_token
        headers = {"Authorization": f"Bearer {token}"}

        endpoints = [
            "/api/analytics/overview",
            "/api/analytics/bmi-trends",
            "/api/analytics/timeline?limit=10&offset=0",
            "/api/assessments?page=1&page_size=10",
            "/api/insights",
            "/api/insights/summary",
        ]

        for ep in endpoints:
            latencies = []
            for _ in range(10):
                t0 = time.perf_counter()
                res = client.get(ep, headers=headers)
                t1 = time.perf_counter()
                assert res.status_code == 200
                latencies.append((t1 - t0) * 1000)

            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            p95 = latencies[int(len(latencies) * 0.95)]
            print(f"[BENCHMARK] {ep}: Mean={mean_lat:.2f}ms, P95={p95:.2f}ms")
            assert mean_lat < 150.0, f"{ep} mean latency ({mean_lat:.2f}ms) exceeded threshold"

    def test_concurrent_authenticated_requests(self, client, test_user_with_token, populate_test_assessments):
        _, token = test_user_with_token
        headers = {"Authorization": f"Bearer {token}"}

        def make_request(idx):
            ep = "/api/analytics/overview" if idx % 2 == 0 else "/api/insights"
            t0 = time.perf_counter()
            res = client.get(ep, headers=headers)
            t1 = time.perf_counter()
            return res.status_code, (t1 - t0) * 1000

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(make_request, range(20)))

        for status_code, lat in results:
            assert status_code == 200

        concurrent_latencies = [lat for _, lat in results]
        mean_conc = sum(concurrent_latencies) / len(concurrent_latencies)
        p95_conc = sorted(concurrent_latencies)[int(len(concurrent_latencies) * 0.95)]
        print(f"[BENCHMARK] Concurrent (20 reqs, 8 threads): Mean={mean_conc:.2f}ms, P95={p95_conc:.2f}ms")
        assert mean_conc < 200.0
