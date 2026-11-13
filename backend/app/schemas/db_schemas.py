"""
backend/app/schemas/db_schemas.py
===================================
Pydantic schemas for database-backed API responses.
Extends the existing prediction schemas without modifying them.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Assessment schemas
# ---------------------------------------------------------------------------

class AssessmentCreate(BaseModel):
    """21 CDC features accepted on creation — same validation as DiabetesFeatureInput."""
    HighBP: int = Field(..., ge=0, le=1)
    HighChol: int = Field(..., ge=0, le=1)
    CholCheck: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=10.0, le=100.0)
    Smoker: int = Field(..., ge=0, le=1)
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    Fruits: int = Field(..., ge=0, le=1)
    Veggies: int = Field(..., ge=0, le=1)
    HvyAlcoholConsump: int = Field(..., ge=0, le=1)
    AnyHealthcare: int = Field(..., ge=0, le=1)
    NoDocbcCost: int = Field(..., ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5)
    MentHlth: int = Field(..., ge=0, le=30)
    PhysHlth: int = Field(..., ge=0, le=30)
    DiffWalk: int = Field(..., ge=0, le=1)
    Sex: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=1, le=13)
    Education: int = Field(..., ge=1, le=6)
    Income: int = Field(..., ge=1, le=8)
    user_id: Optional[int] = None


class AssessmentOut(BaseModel):
    """Assessment record returned from database."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int
    created_at: datetime


class AssessmentListOut(BaseModel):
    """Paginated assessment list response."""
    total: int
    page: int
    page_size: int
    items: List[AssessmentOut]


# ---------------------------------------------------------------------------
# Prediction schemas
# ---------------------------------------------------------------------------

class PredictionOut(BaseModel):
    """Prediction record returned from database."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    assessment_id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    predicted_class: int
    predicted_label: str
    risk_tier: str
    confidence: float
    probability_no_diabetes: float
    probability_prediabetes: float
    probability_diabetes: float
    model_version: str
    created_at: datetime


class PredictionListOut(BaseModel):
    """Paginated prediction list response."""
    total: int
    page: int
    page_size: int
    items: List[PredictionOut]


# ---------------------------------------------------------------------------
# Combined predict response (inference + persisted IDs)
# ---------------------------------------------------------------------------

class PredictAndSaveResponse(BaseModel):
    """
    Returned by POST /api/predict.
    Combines the ML inference result with the persisted record IDs.
    """
    # Persisted IDs
    assessment_id: int
    prediction_id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None

    # Inference outputs
    predicted_class: int
    predicted_label: str
    risk_tier: str
    confidence: float

    # Three model-estimated probabilities (not clinically calibrated)
    probability_no_diabetes: float
    probability_prediabetes: float
    probability_diabetes: float

    model_version: str
    disclaimer: str


# ---------------------------------------------------------------------------
# Enhanced health response with database status
# ---------------------------------------------------------------------------

class HealthResponseV2(BaseModel):
    """Extended health check including database connectivity."""
    status: str
    timestamp: str
    model_loaded: bool
    database_connected: bool
    version: str


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    """Live database-backed statistics for the Dashboard page."""
    total_assessments: int
    total_predictions: int
    class_distribution: dict  # {0: count, 1: count, 2: count}
    recent_assessments: List[AssessmentOut]

