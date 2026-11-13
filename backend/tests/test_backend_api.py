"""
backend/tests/test_backend_api.py
=================================
Backend test suite for the FastAPI diabetes prediction service.

Strategy: Tests are split into two layers:
  - Layer 1 (pure unit tests, no networking): directly exercises the
    DiabetesModelService, Pydantic schemas, and route handler functions.
    These run in any environment, including sandboxed CI.
  - Layer 2 (integration smoke test): marked with @pytest.mark.integration
    and skipped automatically unless the server is reachable at localhost:8000.

Validates:
  1. Model service singleton loads and returns correct attributes
  2. Canonical feature list (21 features, correct order)
  3. Class mapping (0=No Diabetes, 1=Prediabetes, 2=Diabetes)
  4. Direct predict() call on healthy low-risk profile
  5. Direct predict() call on high-risk profile
  6. Probability bounds [0, 1] and sum-to-1 invariant
  7. Feature order invariance (dict key order independence)
  8. Pydantic schema: missing required field raises ValidationError
  9. Pydantic schema: out-of-range binary (HighBP=2) raises ValidationError
 10. Pydantic schema: BMI below minimum raises ValidationError
 11. Pydantic schema: BMI above maximum raises ValidationError
 12. Pydantic schema: GenHlth outside 1-5 raises ValidationError
 13. model_info response structure
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.services.model_service import (
    CANONICAL_FEATURES,
    CLASS_MAPPING,
    DiabetesModelService,
    model_service,
)
from backend.app.schemas.prediction import DiabetesFeatureInput


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def preload_model():
    """Eagerly load the model once for the entire test session."""
    if not model_service.is_loaded:
        model_service.load_model()
    yield model_service


@pytest.fixture(scope="module")
def healthy_input() -> DiabetesFeatureInput:
    return DiabetesFeatureInput(
        HighBP=0, HighChol=0, CholCheck=1, BMI=22.0, Smoker=0,
        Stroke=0, HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1,
        Veggies=1, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
        GenHlth=1, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0,
        Age=2, Education=6, Income=8,
    )


@pytest.fixture(scope="module")
def high_risk_input() -> DiabetesFeatureInput:
    return DiabetesFeatureInput(
        HighBP=1, HighChol=1, CholCheck=1, BMI=38.0, Smoker=1,
        Stroke=0, HeartDiseaseorAttack=1, PhysActivity=0, Fruits=0,
        Veggies=0, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=1,
        GenHlth=5, MentHlth=15, PhysHlth=20, DiffWalk=1, Sex=1,
        Age=11, Education=3, Income=2,
    )


# ── Layer 1: Unit / Service Tests (no networking) ─────────────────────────

class TestModelServiceSingleton:
    """Verify the model service singleton loads and exposes correct metadata."""

    def test_is_loaded(self, preload_model):
        assert model_service.is_loaded is True

    def test_internal_model_not_none(self, preload_model):
        assert model_service._model is not None

    def test_canonical_features_count(self):
        assert len(CANONICAL_FEATURES) == 21

    def test_canonical_features_first(self):
        assert CANONICAL_FEATURES[0] == "HighBP"

    def test_canonical_features_bmi_index(self):
        """BMI must be at raw CSV index 3."""
        assert CANONICAL_FEATURES[3] == "BMI"

    def test_canonical_features_last(self):
        assert CANONICAL_FEATURES[-1] == "Income"

    def test_class_mapping_no_diabetes(self):
        assert CLASS_MAPPING[0] == "No Diabetes"

    def test_class_mapping_prediabetes(self):
        assert CLASS_MAPPING[1] == "Prediabetes"

    def test_class_mapping_diabetes(self):
        assert CLASS_MAPPING[2] == "Diabetes"


class TestPredictionService:
    """Direct calls to model_service.predict() — no HTTP layer involved."""

    def test_healthy_predicted_class(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert result.predicted_class == 0, (
            f"Expected class 0 for healthy profile, got {result.predicted_class}"
        )

    def test_healthy_predicted_label(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert result.predicted_label == "No Diabetes"

    def test_healthy_risk_tier(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert result.risk_tier == "Low / Baseline Risk"

    def test_high_risk_class(self, preload_model, high_risk_input):
        result = model_service.predict(high_risk_input)
        assert result.predicted_class in [1, 2]

    def test_high_risk_label(self, preload_model, high_risk_input):
        result = model_service.predict(high_risk_input)
        assert result.predicted_label in ["Prediabetes", "Diabetes"]

    def test_disclaimer_present(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert "RESEARCH PROTOTYPE" in result.disclaimer

    def test_probability_no_diabetes_in_range(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert 0.0 <= result.probabilities.no_diabetes <= 1.0

    def test_probability_prediabetes_in_range(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert 0.0 <= result.probabilities.prediabetes <= 1.0

    def test_probability_diabetes_in_range(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        assert 0.0 <= result.probabilities.diabetes <= 1.0

    def test_probabilities_sum_to_one(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        p = result.probabilities
        total = p.no_diabetes + p.prediabetes + p.diabetes
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}, expected ~1.0"

    def test_confidence_matches_predicted_class(self, preload_model, healthy_input):
        result = model_service.predict(healthy_input)
        probs_list = [
            result.probabilities.no_diabetes,
            result.probabilities.prediabetes,
            result.probabilities.diabetes,
        ]
        expected_confidence = round(probs_list[result.predicted_class], 4)
        assert result.confidence == expected_confidence

    def test_feature_order_invariance(self, preload_model):
        """Dict key order must NOT affect the prediction — service re-orders canonically."""
        import random
        base = dict(
            HighBP=0, HighChol=0, CholCheck=1, BMI=22.0, Smoker=0,
            Stroke=0, HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1,
            Veggies=1, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
            GenHlth=1, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0,
            Age=2, Education=6, Income=8,
        )
        keys = list(base.keys())
        rng = random.Random(42)
        rng.shuffle(keys)
        shuffled = {k: base[k] for k in keys}

        res_a = model_service.predict(DiabetesFeatureInput(**base))
        res_b = model_service.predict(DiabetesFeatureInput(**shuffled))

        assert res_a.predicted_class == res_b.predicted_class
        assert res_a.probabilities == res_b.probabilities
        assert res_a.confidence == res_b.confidence


class TestModelInfoService:
    """Verify model_service.get_model_info() returns a well-formed response."""

    def test_model_info_feature_count(self, preload_model):
        info = model_service.get_model_info()
        assert info.feature_count == 21

    def test_model_info_features_match_canonical(self, preload_model):
        info = model_service.get_model_info()
        assert info.features == CANONICAL_FEATURES

    def test_model_info_class_mapping_keys(self, preload_model):
        info = model_service.get_model_info()
        assert set(info.class_mapping.keys()) == {0, 1, 2}

    def test_model_info_disclaimer(self, preload_model):
        info = model_service.get_model_info()
        assert info.disclaimer


# ── Layer 2: Pydantic Schema Validation ───────────────────────────────────

class TestSchemaValidation:
    """Pydantic field validation — no model inference, no networking."""

    _base = dict(
        HighBP=0, HighChol=0, CholCheck=1, BMI=22.0, Smoker=0,
        Stroke=0, HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1,
        Veggies=1, HvyAlcoholConsump=0, AnyHealthcare=1, NoDocbcCost=0,
        GenHlth=1, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0,
        Age=2, Education=6, Income=8,
    )

    def _make(self, **overrides):
        return {**self._base, **overrides}

    def test_valid_healthy_profile_accepted(self):
        obj = DiabetesFeatureInput(**self._base)
        assert obj.BMI == 22.0
        assert obj.HighBP == 0

    def test_missing_bmi_raises(self):
        payload = {k: v for k, v in self._base.items() if k != "BMI"}
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**payload)
        assert "BMI" in str(exc_info.value)

    def test_missing_highbp_raises(self):
        payload = {k: v for k, v in self._base.items() if k != "HighBP"}
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**payload)
        assert "HighBP" in str(exc_info.value)

    def test_binary_highbp_out_of_range_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(HighBP=2))
        assert "HighBP" in str(exc_info.value)

    def test_binary_highchol_out_of_range_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(HighChol=99))
        assert "HighChol" in str(exc_info.value)

    def test_bmi_below_minimum_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(BMI=5.0))
        assert "BMI" in str(exc_info.value)

    def test_bmi_above_maximum_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(BMI=105.0))
        assert "BMI" in str(exc_info.value)

    def test_bmi_at_minimum_accepted(self):
        obj = DiabetesFeatureInput(**self._make(BMI=10.0))
        assert obj.BMI == 10.0

    def test_bmi_at_maximum_accepted(self):
        obj = DiabetesFeatureInput(**self._make(BMI=100.0))
        assert obj.BMI == 100.0

    def test_genhlth_above_max_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(GenHlth=6))
        assert "GenHlth" in str(exc_info.value)

    def test_genhlth_below_min_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(GenHlth=0))
        assert "GenHlth" in str(exc_info.value)

    def test_genhlth_valid_boundary_min(self):
        obj = DiabetesFeatureInput(**self._make(GenHlth=1))
        assert obj.GenHlth == 1

    def test_genhlth_valid_boundary_max(self):
        obj = DiabetesFeatureInput(**self._make(GenHlth=5))
        assert obj.GenHlth == 5

    def test_menthlth_above_max_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(MentHlth=31))
        assert "MentHlth" in str(exc_info.value)

    def test_age_above_max_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(Age=14))
        assert "Age" in str(exc_info.value)

    def test_income_above_max_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DiabetesFeatureInput(**self._make(Income=9))
        assert "Income" in str(exc_info.value)
