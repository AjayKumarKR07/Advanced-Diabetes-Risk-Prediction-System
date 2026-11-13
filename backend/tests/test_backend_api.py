"""
backend/tests/test_backend_api.py
=================================
Comprehensive backend test suite for FastAPI diabetes prediction service.

Validates:
  1. Health check endpoint (GET /api/health)
  2. Model info endpoint (GET /api/model-info)
  3. Valid prediction execution (POST /api/predict)
  4. Probability distribution properties (3 classes, sum to 1.0, bounded [0, 1])
  5. Correct class mappings (0: No Diabetes, 1: Prediabetes, 2: Diabetes)
  6. Rejection of invalid / out-of-range inputs (BMI, HighBP, GenHlth)
  7. Rejection of incomplete payloads (missing required features)
  8. Feature order invariance (dictionary key order independence)
  9. Structured validation error response format
 10. Direct model service singleton loading
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_service import CANONICAL_FEATURES, CLASS_MAPPING, model_service
from backend.app.schemas.prediction import DiabetesFeatureInput


@pytest.fixture(scope="module")
def client():
    """Create test client with startup/lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_healthy_payload() -> dict:
    return {
        "HighBP": 0,
        "HighChol": 0,
        "CholCheck": 1,
        "BMI": 22.0,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 1,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 0,
        "Age": 2,
        "Education": 6,
        "Income": 8,
    }


@pytest.fixture
def valid_high_risk_payload() -> dict:
    return {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 38.0,
        "Smoker": 1,
        "Stroke": 0,
        "HeartDiseaseorAttack": 1,
        "PhysActivity": 0,
        "Fruits": 0,
        "Veggies": 0,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 1,
        "GenHlth": 5,
        "MentHlth": 15,
        "PhysHlth": 20,
        "DiffWalk": 1,
        "Sex": 1,
        "Age": 11,
        "Education": 3,
        "Income": 2,
    }


# ── Health & Root Tests ────────────────────────────────────────────────────

def test_root_endpoint(client):
    """Verify root endpoint returns service info, status, and disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert "documentation" in data
    assert "disclaimer" in data


def test_health_endpoint(client):
    """Verify GET /api/health returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "timestamp" in data
    assert "version" in data


# ── Model Info Tests ───────────────────────────────────────────────────────

def test_model_info_endpoint(client):
    """Verify GET /api/model-info returns complete architecture and evaluation metadata."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 21
    assert data["features"] == CANONICAL_FEATURES
    assert data["class_mapping"]["0"] == "No Diabetes"
    assert data["class_mapping"]["1"] == "Prediabetes"
    assert data["class_mapping"]["2"] == "Diabetes"
    assert "evaluation_performance" in data
    assert "disclaimer" in data


# ── Prediction Endpoint Tests ──────────────────────────────────────────────

def test_predict_healthy_input(client, valid_healthy_payload):
    """Verify prediction on low-risk healthy profile."""
    response = client.post("/api/predict", json=valid_healthy_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] == 0
    assert data["predicted_label"] == "No Diabetes"
    assert data["risk_tier"] == "Low / Baseline Risk"

    # Probabilities check
    probs = data["probabilities"]
    assert probs["no_diabetes"] > probs["diabetes"]
    assert pytest.approx(probs["no_diabetes"] + probs["prediabetes"] + probs["diabetes"], abs=1e-3) == 1.0
    assert "RESEARCH PROTOTYPE" in data["disclaimer"]


def test_predict_high_risk_input(client, valid_high_risk_payload):
    """Verify prediction on elevated metabolic risk profile."""
    response = client.post("/api/predict", json=valid_high_risk_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [1, 2]
    assert data["predicted_label"] in ["Prediabetes", "Diabetes"]
    assert "Risk" in data["risk_tier"]

    probs = data["probabilities"]
    prob_sum = probs["no_diabetes"] + probs["prediabetes"] + probs["diabetes"]
    assert pytest.approx(prob_sum, abs=1e-3) == 1.0


def test_probability_bounds_and_structure(client, valid_healthy_payload):
    """Verify every class probability is strictly bounded in [0, 1]."""
    response = client.post("/api/predict", json=valid_healthy_payload)
    assert response.status_code == 200
    probs = response.json()["probabilities"]

    for key in ["no_diabetes", "prediabetes", "diabetes"]:
        val = probs[key]
        assert 0.0 <= val <= 1.0, f"Probability {key} out of bounds: {val}"


# ── Validation & Error Handling Tests ──────────────────────────────────────

def test_reject_missing_feature(client, valid_healthy_payload):
    """Verify 422 Unprocessable Entity when required feature is omitted."""
    incomplete = valid_healthy_payload.copy()
    del incomplete["BMI"]

    response = client.post("/api/predict", json=incomplete)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert any("BMI" in d["field"] for d in data["details"])


def test_reject_out_of_range_binary(client, valid_healthy_payload):
    """Verify rejection when binary feature exceeds {0, 1}."""
    invalid = valid_healthy_payload.copy()
    invalid["HighBP"] = 2  # Must be 0 or 1

    response = client.post("/api/predict", json=invalid)
    assert response.status_code == 422
    data = response.json()
    assert any("HighBP" in d["field"] for d in data["details"])


def test_reject_out_of_range_bmi_low(client, valid_healthy_payload):
    """Verify rejection when BMI is below plausible minimum (10.0)."""
    invalid = valid_healthy_payload.copy()
    invalid["BMI"] = 5.0

    response = client.post("/api/predict", json=invalid)
    assert response.status_code == 422
    data = response.json()
    assert any("BMI" in d["field"] for d in data["details"])


def test_reject_out_of_range_bmi_high(client, valid_healthy_payload):
    """Verify rejection when BMI exceeds maximum bound (100.0)."""
    invalid = valid_healthy_payload.copy()
    invalid["BMI"] = 105.0

    response = client.post("/api/predict", json=invalid)
    assert response.status_code == 422
    data = response.json()
    assert any("BMI" in d["field"] for d in data["details"])


def test_reject_out_of_range_genhlth(client, valid_healthy_payload):
    """Verify rejection when GenHlth is outside 1-5 scale."""
    invalid = valid_healthy_payload.copy()
    invalid["GenHlth"] = 6

    response = client.post("/api/predict", json=invalid)
    assert response.status_code == 422
    data = response.json()
    assert any("GenHlth" in d["field"] for d in data["details"])


def test_feature_order_invariance(client, valid_healthy_payload):
    """
    Verify that shuffling dictionary key order does not alter predicted probabilities,
    proving backend canonical re-ordering operates reliably.
    """
    import random
    keys = list(valid_healthy_payload.keys())
    shuffled_keys = keys.copy()
    random.seed(42)
    random.shuffle(shuffled_keys)
    shuffled_payload = {k: valid_healthy_payload[k] for k in shuffled_keys}

    res_orig = client.post("/api/predict", json=valid_healthy_payload).json()
    res_shuf = client.post("/api/predict", json=shuffled_payload).json()

    assert res_orig["predicted_class"] == res_shuf["predicted_class"]
    assert res_orig["probabilities"] == res_shuf["probabilities"]
    assert res_orig["confidence"] == res_shuf["confidence"]


# ── Model Service Unit Tests ───────────────────────────────────────────────

def test_model_service_singleton():
    """Verify unit-level model loading and class attribute verification."""
    assert model_service.is_loaded is True
    assert model_service._model is not None
    assert len(CANONICAL_FEATURES) == 21
    for c_id, c_name in [(0, "No Diabetes"), (1, "Prediabetes"), (2, "Diabetes")]:
        assert CLASS_MAPPING[c_id] == c_name

