"""
backend/app/api/analytics_routes.py
===================================
FastAPI endpoints for personal health analytics, longitudinal BMI and health
trends, prediction distribution, indicator comparison, and assessment timeline (Phase 6.15).
Every endpoint is strictly scoped to the authenticated user via get_current_user.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.schemas.analytics_schemas import (
    AnalyticsOverviewResponse,
    BmiTrendsResponse,
    HealthTrendsResponse,
    IndicatorComparisonResponse,
    PredictionDistributionResponse,
    TimelineResponse,
)
from backend.app.services import analytics_service

analytics_router = APIRouter(prefix="/analytics", tags=["Personal Health Analytics"])


@analytics_router.get(
    "/overview",
    response_model=AnalyticsOverviewResponse,
    status_code=status.HTTP_200_OK,
    summary="User Personal Health Analytics Overview",
    description="Returns authenticated user's KPI metrics, latest evaluation summary, and trend statistics.",
)
async def get_analytics_overview(
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalyticsOverviewResponse:
    """Retrieve comprehensive personal KPI analytics for the authenticated user."""
    return analytics_service.get_analytics_overview(
        db, user_id=current_user.id, start_date=start_date, end_date=end_date
    )


@analytics_router.get(
    "/prediction-distribution",
    response_model=PredictionDistributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prediction Class Distribution",
    description="Returns breakdown of No Diabetes, Prediabetes, and Diabetes model classifications.",
)
async def get_prediction_distribution(
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionDistributionResponse:
    """Retrieve prediction outcome count and percentage distribution for authenticated user."""
    return analytics_service.get_prediction_distribution(
        db, user_id=current_user.id, start_date=start_date, end_date=end_date
    )


@analytics_router.get(
    "/bmi-trends",
    response_model=BmiTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Longitudinal BMI Trajectory",
    description="Returns chronological BMI assessment points and change summary for authenticated user.",
)
async def get_bmi_trends(
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> BmiTrendsResponse:
    """Retrieve longitudinal BMI trajectory points and summary."""
    return analytics_service.get_bmi_trends(
        db, user_id=current_user.id, start_date=start_date, end_date=end_date
    )


@analytics_router.get(
    "/health-trends",
    response_model=HealthTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="General Health and Well-being Trends",
    description="Returns chronological self-reported General Health, PhysHlth, and MentHlth ratings.",
)
async def get_health_trends(
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HealthTrendsResponse:
    """Retrieve chronological General Health survey ratings."""
    return analytics_service.get_health_trends(
        db, user_id=current_user.id, start_date=start_date, end_date=end_date
    )


@analytics_router.get(
    "/indicators",
    response_model=IndicatorComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Indicator Longitudinal Comparison",
    description="Returns historical submitted values for 9 key clinical and lifestyle indicators.",
)
async def get_indicator_comparison(
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IndicatorComparisonResponse:
    """Retrieve multi-assessment comparison table for key health indicators."""
    return analytics_service.get_indicator_comparison(
        db, user_id=current_user.id, start_date=start_date, end_date=end_date
    )


@analytics_router.get(
    "/timeline",
    response_model=TimelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Assessment Activity Timeline",
    description="Returns paginated assessment timeline with model predictions and indicators.",
)
async def get_assessment_timeline(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Records per page"),
    start_date: Optional[str] = Query(None, description="Start date ISO string filter"),
    end_date: Optional[str] = Query(None, description="End date ISO string filter"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TimelineResponse:
    """Retrieve paginated assessment activity timeline for authenticated user."""
    return analytics_service.get_assessment_timeline(
        db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        start_date=start_date,
        end_date=end_date,
    )
