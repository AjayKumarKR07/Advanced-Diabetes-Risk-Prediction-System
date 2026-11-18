"""
backend/app/api/explain_routes.py
===================================
XAI Feature Contribution Breakdown endpoint.
Computes logistic regression coefficient × standardized value contributions
for the top 5 risk factors driving each prediction.
Educational only — not causal inference.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.services.model_service import model_service, CANONICAL_FEATURES, CLASS_MAPPING

logger = logging.getLogger(__name__)

explain_router = APIRouter(prefix="/explain", tags=["Explainable AI"])

FEATURE_LABELS = {
    "HighBP":              "High Blood Pressure",
    "HighChol":            "High Cholesterol",
    "CholCheck":           "Cholesterol Check",
    "BMI":                 "Body Mass Index",
    "Smoker":              "Smoker",
    "Stroke":              "Stroke History",
    "HeartDiseaseorAttack":"Heart Disease / Attack",
    "PhysActivity":        "Physical Activity",
    "Fruits":              "Fruit Consumption",
    "Veggies":             "Vegetable Consumption",
    "HvyAlcoholConsump":   "Heavy Alcohol Use",
    "AnyHealthcare":       "Healthcare Access",
    "NoDocbcCost":         "No Doctor (Cost Barrier)",
    "GenHlth":             "General Health (Self-rated)",
    "MentHlth":            "Mental Health Days (past 30)",
    "PhysHlth":            "Physical Health Days (past 30)",
    "DiffWalk":            "Difficulty Walking",
    "Sex":                 "Sex",
    "Age":                 "Age Category",
    "Education":           "Education Level",
    "Income":              "Income Level",
}

VALUE_FORMATS = {
    "HighBP":               lambda v: "Yes" if v == 1 else "No",
    "HighChol":             lambda v: "Yes" if v == 1 else "No",
    "CholCheck":            lambda v: "Yes" if v == 1 else "No",
    "BMI":                  lambda v: f"{v:.1f}",
    "Smoker":               lambda v: "Yes" if v == 1 else "No",
    "Stroke":               lambda v: "Yes" if v == 1 else "No",
    "HeartDiseaseorAttack": lambda v: "Yes" if v == 1 else "No",
    "PhysActivity":         lambda v: "Yes" if v == 1 else "No",
    "Fruits":               lambda v: "Yes" if v == 1 else "No",
    "Veggies":              lambda v: "Yes" if v == 1 else "No",
    "HvyAlcoholConsump":    lambda v: "Yes" if v == 1 else "No",
    "AnyHealthcare":        lambda v: "Yes" if v == 1 else "No",
    "NoDocbcCost":          lambda v: "Yes" if v == 1 else "No",
    "GenHlth":              lambda v: ["Excellent","Very Good","Good","Fair","Poor"][int(v)-1] if 1<=v<=5 else str(v),
    "MentHlth":             lambda v: f"{int(v)} days",
    "PhysHlth":             lambda v: f"{int(v)} days",
    "DiffWalk":             lambda v: "Yes" if v == 1 else "No",
    "Sex":                  lambda v: "Male" if v == 1 else "Female",
    "Age":                  lambda v: f"Category {int(v)} ({13*int(v)+5}-{13*int(v)+17}y approx.)" if v else str(v),
    "Education":            lambda v: f"Level {int(v)}",
    "Income":               lambda v: f"Level {int(v)}",
}


class FeatureContribution(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str         # "risk_increasing" | "risk_reducing" | "neutral"
    value: float
    value_label: str
    abs_contribution: float


class ExplainResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    top_contributors: List[FeatureContribution]
    all_contributions: List[FeatureContribution]
    disclaimer: str = (
        "Feature contributions are computed from logistic regression coefficients × "
        "standardized input values for educational awareness only. "
        "They indicate directional associations in the training data — NOT causal relationships. "
        "This is a research prototype; not a clinical diagnostic tool."
    )


class ExplainRequest(BaseModel):
    features: dict
    predicted_class: Optional[int] = None


def _compute_contributions(features_dict: dict, predicted_class: int) -> List[FeatureContribution]:
    """Compute signed feature contributions for the given predicted class."""
    import numpy as np
    import pandas as pd

    if not model_service.is_loaded:
        model_service.load_model()

    pipeline = model_service._model

    # Build input array
    raw_values = [float(features_dict.get(f, 0.0)) for f in CANONICAL_FEATURES]
    df = pd.DataFrame([raw_values], columns=CANONICAL_FEATURES)

    # Try to extract scaler and classifier from pipeline
    scaler = None
    clf = None
    for name, step in pipeline.steps:
        if hasattr(step, 'transform') and hasattr(step, 'mean_'):
            scaler = step
        if hasattr(step, 'coef_'):
            clf = step

    if clf is None:
        raise ValueError("Could not find logistic regression classifier in pipeline.")

    # Get standardized values
    if scaler is not None:
        try:
            scaled = scaler.transform(df)[0]
        except Exception:
            scaled = raw_values
    else:
        scaled = raw_values

    # Coefficients shape: (n_classes, n_features) for multinomial, or (1, n_features) for binary
    coefs = clf.coef_
    if coefs.shape[0] == 1:
        # Binary case — use class 1 coefficients
        class_coef = coefs[0]
    else:
        # Multinomial — use the predicted class row
        class_idx = min(predicted_class, coefs.shape[0] - 1)
        class_coef = coefs[class_idx]

    contributions = []
    for i, feat in enumerate(CANONICAL_FEATURES):
        raw_val = raw_values[i]
        contrib = float(class_coef[i]) * float(scaled[i])
        abs_c = abs(contrib)

        if abs_c < 1e-6:
            direction = "neutral"
        elif contrib > 0:
            direction = "risk_increasing"
        else:
            direction = "risk_reducing"

        fmt_fn = VALUE_FORMATS.get(feat)
        value_label = fmt_fn(raw_val) if fmt_fn else str(raw_val)

        contributions.append(FeatureContribution(
            feature=feat,
            label=FEATURE_LABELS.get(feat, feat),
            contribution=round(contrib, 4),
            abs_contribution=round(abs_c, 4),
            direction=direction,
            value=raw_val,
            value_label=value_label,
        ))

    # Sort by absolute contribution descending
    contributions.sort(key=lambda x: x.abs_contribution, reverse=True)
    return contributions


@explain_router.post("", response_model=ExplainResponse)
async def explain_prediction(
    payload: ExplainRequest,
    current_user: User = Depends(get_current_user),
) -> ExplainResponse:
    """
    Compute XAI feature contribution breakdown for a diabetes risk prediction.
    Returns top 5 factors (by absolute contribution) for the predicted class.
    """
    try:
        if not model_service.is_loaded:
            model_service.load_model()

        features = payload.features
        # Run predict to get the class if not provided
        if payload.predicted_class is None:
            from backend.app.schemas.prediction import DiabetesFeatureInput
            feat_input = DiabetesFeatureInput(**features)
            result = model_service.predict(feat_input)
            predicted_class = result.predicted_class
        else:
            predicted_class = payload.predicted_class

        all_contribs = _compute_contributions(features, predicted_class)
        top5 = all_contribs[:5]

        return ExplainResponse(
            predicted_class=predicted_class,
            predicted_label=CLASS_MAPPING.get(predicted_class, "Unknown"),
            top_contributors=top5,
            all_contributions=all_contribs,
        )

    except Exception as exc:
        logger.error("XAI explain error for user %s: %s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not compute feature contributions: {str(exc)}",
        )
