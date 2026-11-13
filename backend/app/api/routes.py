"""
backend/app/api/routes.py
=========================
FastAPI API router for diabetes prediction, model inspection, and health checking.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    DiabetesFeatureInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from backend.app.services.model_service import model_service

api_router = APIRouter()


@api_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns backend server health status and model artifact loading verification.",
)
async def health_check() -> HealthResponse:
    """Verify backend health and model loading readiness."""
    try:
        if not model_service.is_loaded:
            model_service.load_model()
        is_ready = model_service.is_loaded
    except Exception:
        is_ready = False

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=is_ready,
        version=settings.VERSION,
    )


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


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate 3-class diabetes risk from 21 CDC health indicators",
    description=(
        "Executes forward inference using the evaluated Phase 4.2 Logistic Regression pipeline. "
        "Strictly an academic research prototype; NOT a clinical diagnostic tool."
    ),
)
async def predict_diabetes_risk(payload: DiabetesFeatureInput) -> PredictionResponse:
    """
    Validate input features, enforce canonical order, and return 3-class prediction with probabilities.
    """
    try:
        return model_service.predict(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(exc)}",
        )

