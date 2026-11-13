"""
backend/app/api/routes.py
=========================
FastAPI API router — diabetes prediction, model inspection, health check,
and DB-backed assessment/prediction history endpoints (Phase 6.6).
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.api.auth_routes import auth_router
from backend.app.core.config import settings
from backend.app.core.database import check_db_connection, get_db
from backend.app.core.security import get_current_user, get_optional_user
from backend.app.models.user import User
from backend.app.schemas.db_schemas import (
    AssessmentCreate,
    AssessmentListOut,
    AssessmentOut,
    DashboardStats,
    HealthResponseV2,
    PredictAndSaveResponse,
    PredictionListOut,
    PredictionOut,
)
from backend.app.schemas.prediction import ModelInfoResponse
from backend.app.services import db_service
from backend.app.services.model_service import model_service

api_router = APIRouter()

# Mount authentication routes (/api/auth/register, /api/auth/login, /api/auth/me)
api_router.include_router(auth_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@api_router.get(
    "/health",
    response_model=HealthResponseV2,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns backend health, model loading status, and database connectivity.",
)
async def health_check() -> HealthResponseV2:
    """Verify backend health, model readiness, and database connectivity."""
    try:
        if not model_service.is_loaded:
            model_service.load_model()
        is_ready = model_service.is_loaded
    except Exception:
        is_ready = False

    db_connected = check_db_connection()

    return HealthResponseV2(
        status="healthy" if is_ready else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=is_ready,
        database_connected=db_connected,
        version=settings.VERSION,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@api_router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model metadata and performance catalog",
    description="Returns architecture details, required input features, and Phase 5 evaluated metrics.",
)
async def get_model_info() -> ModelInfoResponse:
    """Retrieve model metadata, feature catalogue, and Phase 5 evaluation benchmarks."""
    try:
        return model_service.get_model_info()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model metadata: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Predict — inference + persist to DB
# ---------------------------------------------------------------------------

@api_router.post(
    "/predict",
    response_model=PredictAndSaveResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate 3-class diabetes risk from 21 CDC health indicators",
    description=(
        "Executes forward inference using the evaluated Phase 4.2 Logistic Regression pipeline "
        "and persists the assessment and prediction to PostgreSQL. "
        "Strictly an academic research prototype; NOT a clinical diagnostic tool."
    ),
)
async def predict_diabetes_risk(
    payload: AssessmentCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> PredictAndSaveResponse:
    """
    Validate input features, run ML inference, persist assessment + prediction,
    and return a combined response with persisted IDs.
    """
    # If user is authenticated and payload.user_id was not explicitly specified, associate with current user
    if payload.user_id is None and current_user is not None:
        payload.user_id = current_user.id
    # 1. Run ML inference (uses existing model_service — model is frozen)
    try:
        from backend.app.schemas.prediction import DiabetesFeatureInput

        feature_input = DiabetesFeatureInput(**payload.model_dump(exclude={"user_id"}))
        inference_result = model_service.predict(feature_input)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}",
        )

    # 2. Persist assessment + prediction atomically
    try:
        assessment = db_service.create_assessment(db, payload)

        prediction = db_service.create_prediction(
            db=db,
            assessment_id=assessment.id,
            predicted_class=inference_result.predicted_class,
            predicted_label=inference_result.predicted_label,
            risk_tier=inference_result.risk_tier,
            confidence=inference_result.confidence,
            prob_no_diabetes=inference_result.probabilities.no_diabetes,
            prob_prediabetes=inference_result.probabilities.prediabetes,
            prob_diabetes=inference_result.probabilities.diabetes,
            model_version=inference_result.model_version,
        )

        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database persistence error: {str(exc)}",
        )

    return PredictAndSaveResponse(
        assessment_id=assessment.id,
        prediction_id=prediction.id,
        user_id=assessment.user_id,
        user_name=current_user.full_name if current_user else None,
        predicted_class=inference_result.predicted_class,
        predicted_label=inference_result.predicted_label,
        risk_tier=inference_result.risk_tier,
        confidence=inference_result.confidence,
        probability_no_diabetes=inference_result.probabilities.no_diabetes,
        probability_prediabetes=inference_result.probabilities.prediabetes,
        probability_diabetes=inference_result.probabilities.diabetes,
        model_version=inference_result.model_version,
        disclaimer=settings.DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Stats — live dashboard data
# ---------------------------------------------------------------------------

@api_router.get(
    "/stats",
    response_model=DashboardStats,
    status_code=status.HTTP_200_OK,
    summary="Live database statistics for the dashboard",
)
async def get_stats(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> DashboardStats:
    """Return aggregate counts and class distribution from the database (user-filtered when authenticated)."""
    try:
        user_id = current_user.id if current_user is not None else None
        data = db_service.get_dashboard_stats(db, user_id=user_id)
        return DashboardStats(**data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Assessment history
# ---------------------------------------------------------------------------

@api_router.get(
    "/assessments",
    response_model=AssessmentListOut,
    status_code=status.HTTP_200_OK,
    summary="List paginated assessment records",
)
async def list_assessments(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Records per page"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> AssessmentListOut:
    user_id = current_user.id if current_user is not None else None
    total, items = db_service.list_assessments(db, page=page, page_size=page_size, user_id=user_id)
    return AssessmentListOut(total=total, page=page, page_size=page_size, items=items)


@api_router.get(
    "/assessments/{assessment_id}",
    response_model=AssessmentOut,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a single assessment by ID",
)
async def get_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> AssessmentOut:
    record = db_service.get_assessment(db, assessment_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found.",
        )
    # Enforce ownership check: prevent cross-user access
    if record.user_id is not None and current_user is not None:
        if record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not have permission to view this assessment.",
            )
    return record


@api_router.get(
    "/assessments/{assessment_id}/prediction",
    response_model=PredictionOut,
    status_code=status.HTTP_200_OK,
    summary="Retrieve prediction record associated with an assessment",
)
async def get_assessment_prediction(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> PredictionOut:
    pred = db_service.get_prediction_by_assessment(db, assessment_id)
    if pred is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction for assessment {assessment_id} not found.",
        )
    if pred.assessment and pred.assessment.user_id is not None and current_user is not None:
        if pred.assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not have permission to view this prediction.",
            )
    return pred


@api_router.delete(
    "/assessments/{assessment_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an assessment and its associated prediction",
)
async def delete_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Permanently delete an assessment and its associated prediction record.
    Enforces JWT authentication and strict record ownership.
    """
    record = db_service.get_assessment(db, assessment_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment {assessment_id} not found.",
        )
    if record.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to delete this assessment.",
        )
    try:
        db_service.delete_assessment(db, assessment_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete assessment: {str(exc)}",
        )
    return {
        "message": f"Assessment {assessment_id} and its associated prediction have been deleted.",
        "assessment_id": assessment_id,
    }


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

@api_router.get(
    "/predictions",
    response_model=PredictionListOut,
    status_code=status.HTTP_200_OK,
    summary="List paginated prediction records",
)
async def list_predictions(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Records per page"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> PredictionListOut:
    user_id = current_user.id if current_user is not None else None
    total, items = db_service.list_predictions(db, page=page, page_size=page_size, user_id=user_id)
    return PredictionListOut(total=total, page=page, page_size=page_size, items=items)


@api_router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionOut,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a single prediction by ID",
)
async def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
) -> PredictionOut:
    record = db_service.get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )
    # Enforce ownership check: prevent cross-user access
    if record.assessment and record.assessment.user_id is not None and current_user is not None:
        if record.assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not have permission to view this prediction.",
            )
    return record
