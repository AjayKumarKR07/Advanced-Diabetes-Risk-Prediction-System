"""
backend/app/api/insights_routes.py
==================================
FastAPI endpoints for intelligent, explainable, educational health insights
and personalized non-medical recommendations (Phase 6.16).
All routes require authentication and are strictly scoped to current_user.id.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.schemas.insights_schemas import (
    HealthInsightsResponse,
    InsightsHistoryResponse,
    InsightsSummary,
)
from backend.app.services import insights_service

insights_router = APIRouter(prefix="/insights", tags=["Intelligent Health Insights"])


@insights_router.get(
    "",
    response_model=HealthInsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Personalized Educational Health Insights",
    description="Returns transparent, rule-based observations, educational suggestions, and historical comparisons.",
)
async def get_insights(
    assessment_id: Optional[int] = Query(None, description="Optional specific assessment ID to analyze"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HealthInsightsResponse:
    """Retrieve explainable health insights and recommendations for the authenticated user."""
    return insights_service.get_health_insights(
        db, user_id=current_user.id, assessment_id=assessment_id
    )


@insights_router.get(
    "/summary",
    response_model=InsightsSummary,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Health Insights KPI Summary",
    description="Returns aggregate counts of active insights, priority tiers, and category breakdowns.",
)
async def get_insights_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> InsightsSummary:
    """Retrieve high-level summary of active insights for the authenticated user."""
    return insights_service.get_insights_summary(db, user_id=current_user.id)


@insights_router.get(
    "/history",
    response_model=InsightsHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Health Insights Historical Comparisons",
    description="Returns longitudinal comparison items tracking changes across the user's assessments.",
)
async def get_insights_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> InsightsHistoryResponse:
    """Retrieve longitudinal indicator comparison history for the authenticated user."""
    return insights_service.get_insights_history(db, user_id=current_user.id)

