"""
backend/app/api/notifications_routes.py
========================================
Rule-based health notification & assessment reminder center.
Generates smart notifications from existing user assessment data.
No new DB table required — derived from existing assessment/prediction records.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.services import db_service

logger = logging.getLogger(__name__)

notifications_router = APIRouter(prefix="/notifications", tags=["Notifications"])


class Notification(BaseModel):
    id: str
    type: str               # "reminder" | "alert" | "info" | "success"
    title: str
    message: str
    action_label: Optional[str] = None
    action_url: Optional[str] = None
    created_at: str
    is_urgent: bool = False


class NotificationsResponse(BaseModel):
    notifications: List[Notification]
    unread_count: int


def _build_notifications(user: User, db: Session) -> List[Notification]:
    """Generate rule-based notifications from user's assessment history."""
    notifications: List[Notification] = []
    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    try:
        # Fetch user's assessments and predictions
        assessments_result = db_service.get_user_assessments(db, user.id, page=1, page_size=100)

        # Handle both list and paginated responses
        if isinstance(assessments_result, dict):
            assessments = assessments_result.get("items", []) or assessments_result.get("assessments", [])
        elif isinstance(assessments_result, (list, tuple)):
            assessments = list(assessments_result)
        else:
            assessments = []

        total = len(assessments)

    except Exception as e:
        logger.warning("Could not fetch assessments for notifications: %s", e)
        assessments = []
        total = 0

    # 1. Welcome / first assessment prompt
    if total == 0:
        notifications.append(Notification(
            id="first_assessment",
            type="info",
            title="Welcome to DiabetesRisk AI! 👋",
            message="Start your first health assessment to receive personalized risk insights powered by CDC BRFSS data.",
            action_label="Start Assessment",
            action_url="/assess",
            created_at=now_str,
            is_urgent=False,
        ))

    # 2. Second assessment prompt (after first)
    if total == 1:
        notifications.append(Notification(
            id="second_assessment",
            type="reminder",
            title="Track Your Progress",
            message="You have completed 1 assessment. Complete a second evaluation to start tracking your health trends over time.",
            action_label="New Assessment",
            action_url="/assess",
            created_at=now_str,
            is_urgent=False,
        ))

    # 3. 30-day reassessment reminder
    if total > 0 and assessments:
        try:
            latest = assessments[0]
            created = getattr(latest, "created_at", None)
            if created:
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_since = (now - created).days
                if days_since >= 30:
                    notifications.append(Notification(
                        id="reassessment_30d",
                        type="reminder",
                        title="30-Day Reassessment Due",
                        message=f"It has been {days_since} days since your last health assessment. Regular check-ins help track changes in your risk profile.",
                        action_label="Start Reassessment",
                        action_url="/assess",
                        created_at=now_str,
                        is_urgent=days_since >= 60,
                    ))
        except Exception as e:
            logger.debug("Could not compute reassessment reminder: %s", e)

    # 4. Risk-based alerts from latest prediction
    if total > 0 and assessments:
        try:
            latest = assessments[0]
            # Try to get prediction from latest assessment
            predictions = db_service.get_user_predictions(db, user.id, page=1, page_size=1)
            if isinstance(predictions, dict):
                pred_items = predictions.get("items", []) or predictions.get("predictions", [])
            elif isinstance(predictions, (list, tuple)):
                pred_items = list(predictions)
            else:
                pred_items = []

            if pred_items:
                latest_pred = pred_items[0]
                pred_class = getattr(latest_pred, "predicted_class", None)

                if pred_class == 2:  # Diabetes
                    notifications.append(Notification(
                        id="diabetes_alert",
                        type="alert",
                        title="⚠️ Elevated Diabetes Risk Detected",
                        message="Your latest assessment shows an elevated diabetes risk indicator. This is NOT a diagnosis — please consult a qualified healthcare professional for clinical evaluation.",
                        action_label="View Insights",
                        action_url="/insights",
                        created_at=now_str,
                        is_urgent=True,
                    ))
                elif pred_class == 1:  # Prediabetes
                    notifications.append(Notification(
                        id="prediabetes_alert",
                        type="alert",
                        title="Prediabetes Risk Indicator",
                        message="Your assessment indicates prediabetes-range risk factors. Lifestyle changes like regular exercise and balanced nutrition may significantly reduce progression risk.",
                        action_label="View Health Insights",
                        action_url="/insights",
                        created_at=now_str,
                        is_urgent=False,
                    ))
                elif pred_class == 0:  # No Diabetes
                    notifications.append(Notification(
                        id="healthy_badge",
                        type="success",
                        title="✅ Low Risk Profile",
                        message="Your latest assessment shows low diabetes risk indicators. Maintain your healthy lifestyle and schedule regular check-ups.",
                        action_label="View Analytics",
                        action_url="/analytics",
                        created_at=now_str,
                        is_urgent=False,
                    ))
        except Exception as e:
            logger.debug("Could not generate risk-based notification: %s", e)

    # 5. AI Assistant info (always shown)
    notifications.append(Notification(
        id="healthbot_info",
        type="info",
        title="💬 HealthBot is Ready",
        message="Ask HealthBot any question about your results, CDC health indicators, or diabetes risk factors. Click the chat bubble (bottom-right).",
        action_label=None,
        action_url=None,
        created_at=now_str,
        is_urgent=False,
    ))

    # 6. Analytics reminder after 2+ assessments
    if total >= 2:
        notifications.append(Notification(
            id="analytics_available",
            type="success",
            title="📊 Analytics Now Available",
            message=f"You have completed {total} assessments. Visit Analytics to view your BMI trends, risk distribution, and health score progression.",
            action_label="Open Analytics",
            action_url="/analytics",
            created_at=now_str,
            is_urgent=False,
        ))

    return notifications


@notifications_router.get("", response_model=NotificationsResponse)
async def get_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NotificationsResponse:
    """Get smart health notifications for the authenticated user."""
    try:
        notifs = _build_notifications(current_user, db)
        return NotificationsResponse(
            notifications=notifs,
            unread_count=len(notifs),
        )
    except Exception as exc:
        logger.error("Notifications error for user %s: %s", current_user.id, exc)
        # Return empty rather than error — notifications are non-critical
        return NotificationsResponse(notifications=[], unread_count=0)
