"""
backend/app/services/model_service.py
=====================================
Model inference service wrapping the evaluated CDC Logistic Regression pipeline.
Enforces canonical feature ordering, probability extraction, and risk tiering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    DiabetesFeatureInput,
    ModelInfoResponse,
    PredictionResponse,
    ProbabilityDistribution,
)

CLASS_MAPPING = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}

CANONICAL_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


class DiabetesModelService:
    """
    Singleton inference service for the CDC BRFSS 2015 3-class prediction model.
    Loads the evaluated Phase 4.2 Logistic Regression pipeline artifact.
    """
    def __init__(self, model_path: Path = settings.MODEL_PATH) -> None:
        self.model_path = model_path
        self._model: Optional[Any] = None
        self._metadata: Dict[str, Any] = {}
        self._phase5_metrics: Dict[str, Any] = {}

    def load_model(self) -> None:
        """Load and verify the trained scikit-learn Pipeline artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Ensure Phase 4.2 group-aware benchmark completed successfully."
            )

        self._model = joblib.load(self.model_path)

        # Load metadata if present
        if settings.METADATA_PATH.exists():
            try:
                self._metadata = json.loads(settings.METADATA_PATH.read_text())
            except Exception:
                self._metadata = {}

        # Load Phase 5 final evaluation metrics if present
        if settings.PHASE5_METRICS_PATH.exists():
            try:
                self._phase5_metrics = json.loads(settings.PHASE5_METRICS_PATH.read_text())
            except Exception:
                self._phase5_metrics = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _determine_risk_tier(self, probs: np.ndarray, pred_class: int) -> str:
        """
        Assign a clinical research risk categorization based on predicted probabilities.
        Accounts for balanced class weighting probability shifts.
        """
        p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])

        if pred_class == 2 or p2 >= 0.40:
            return "High Risk (Diabetes Indication)"
        elif pred_class == 1 or p1 >= 0.25 or (p1 + p2) >= 0.45:
            return "Elevated Risk (Prediabetes Warning)"
        else:
            return "Low / Baseline Risk"

    def predict(self, input_data: DiabetesFeatureInput) -> PredictionResponse:
        """
        Execute forward-pass inference on validated input features.

        Steps:
        1. Ensure model is loaded.
        2. Construct single-row DataFrame matching canonical feature order.
        3. Call pipeline predict and predict_proba.
        4. Package structured response with probabilities and research disclaimer.
        """
        if self._model is None:
            self.load_model()

        # Construct DataFrame in exact canonical feature order with float64 types
        row_dict = input_data.model_dump()
        df = pd.DataFrame([[float(row_dict[col]) for col in CANONICAL_FEATURES]], columns=CANONICAL_FEATURES)

        # Inference
        raw_pred = self._model.predict(df)
        pred_class = int(raw_pred[0])

        raw_proba = self._model.predict_proba(df)[0]
        # Guarantee non-negative, normalized probabilities
        clipped_proba = np.clip(raw_proba, 0.0, 1.0)
        norm_proba = clipped_proba / clipped_proba.sum()

        p0, p1, p2 = float(norm_proba[0]), float(norm_proba[1]), float(norm_proba[2])

        probabilities = ProbabilityDistribution(
            no_diabetes=round(p0, 4),
            prediabetes=round(p1, 4),
            diabetes=round(p2, 4),
        )

        risk_tier = self._determine_risk_tier(norm_proba, pred_class)
        confidence = round(float(norm_proba[pred_class]), 4)

        return PredictionResponse(
            predicted_class=pred_class,
            predicted_label=CLASS_MAPPING[pred_class],
            probabilities=probabilities,
            risk_tier=risk_tier,
            confidence=confidence,
            disclaimer=settings.DISCLAIMER,
            model_version=self._metadata.get("model_name", "LogisticRegression-GroupAware"),
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return model metadata, feature catalogue, and Phase 5 evaluation performance."""
        if self._model is None:
            self.load_model()

        eval_summary = {}
        if self._phase5_metrics:
            test_m = self._phase5_metrics.get("test_metrics", {})
            eval_summary = {
                "test_accuracy": round(test_m.get("accuracy", 0.6433), 4),
                "balanced_accuracy": round(test_m.get("balanced_accuracy", 0.5172), 4),
                "macro_f1": round(test_m.get("macro_f1", 0.4246), 4),
                "weighted_f1": round(test_m.get("weighted_f1", 0.7187), 4),
                "roc_auc_ovr_macro": round(test_m.get("roc_auc_ovr_macro", 0.7707), 4),
                "class1_prediabetes_recall": round(
                    self._phase5_metrics.get("class1_prediabetes_analysis", {}).get("recall", 0.3089), 4
                ),
                "class1_prediabetes_specificity": round(
                    self._phase5_metrics.get("class1_prediabetes_analysis", {}).get("specificity", 0.8186), 4
                ),
                "test_set_size": self._phase5_metrics.get("split_information", {}).get("test_rows", 50736),
            }

        return ModelInfoResponse(
            model_name="CDC 3-Class Diabetes Risk Predictor",
            model_algorithm="Regularized Logistic Regression (Balanced Class Weighting)",
            dataset="CDC Behavioral Risk Factor Surveillance System (BRFSS 2015, 253,680 rows)",
            feature_count=len(CANONICAL_FEATURES),
            features=CANONICAL_FEATURES,
            class_mapping=CLASS_MAPPING,
            evaluation_performance=eval_summary,
            disclaimer=settings.DISCLAIMER,
        )


# Global service instance
model_service = DiabetesModelService()

