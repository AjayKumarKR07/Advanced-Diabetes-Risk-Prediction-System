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
    type: str               # "assessment" | "reminder" | "info" | "alert" | "success"
    category: str = "informational"  # "assessment" | "informational"
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
    """Generate rule-based notifications from user's actual assessment and prediction history."""
    notifications: List[Notification] = []
    now = datetime.now(timezone.utc)
    now_str = now.isoformat()

    try:
        # Fetch user's assessments scoped strictly to user.id
        _, assessments = db_service.list_assessments(db, page=1, page_size=100, user_id=user.id)
        total_assessments = len(assessments)
    except Exception as e:
        logger.warning("Could not fetch assessments for notifications: %s", e)
        assessments = []
        total_assessments = 0

    try:
        # Fetch user's predictions scoped strictly to user.id
        _, predictions = db_service.list_predictions(db, page=1, page_size=10, user_id=user.id)
    except Exception as e:
        logger.warning("Could not fetch predictions for notifications: %s", e)
        predictions = []

    # 1. Welcome / initial assessment prompt
    if total_assessments == 0:
        notifications.append(Notification(
            id="first_assessment",
            type="info",
            category="informational",
            title="Welcome to DiabetesRisk AI 👋",
            message="Complete your initial health assessment to evaluate 21 CDC BRFSS health indicators with our research model.",
            action_label="Start Assessment",
            action_url="/assess",
            created_at=now_str,
            is_urgent=False,
        ))

    # 2. Longitudinal tracking prompt (after 1 assessment)
    if total_assessments == 1:
        notifications.append(Notification(
            id="second_assessment",
            type="reminder",
            category="informational",
            title="Longitudinal Tracking",
            message="You have recorded 1 assessment. Additional assessments allow the system to analyze longitudinal changes over time.",
            action_label="New Assessment",
            action_url="/assess",
            created_at=now_str,
            is_urgent=False,
        ))

    # 3. 30-day reassessment reminder (based on actual latest assessment date)
    if total_assessments > 0 and assessments:
        try:
            latest_assessment = assessments[0]
            created = getattr(latest_assessment, "created_at", None)
            if created:
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_since = (now - created).days
                if days_since >= 30:
                    notifications.append(Notification(
                        id=f"reassessment_30d_{latest_assessment.id}",
                        type="reminder",
                        category="informational",
                        title="Periodic Reassessment Due",
                        message=f"It has been {days_since} days since your last recorded assessment. Regular check-ins support accurate longitudinal tracking.",
                        action_label="Start Reassessment",
                        action_url="/assess",
                        created_at=now_str,
                        is_urgent=days_since >= 60,
                    ))
        except Exception as e:
            logger.debug("Could not compute reassessment reminder: %s", e)

    # 4. Assessment-triggered notification from saved prediction record
    # Canonical classes: Class 0 (No Diabetes), Class 1 (Prediabetes), Class 2 (Diabetes)
    # Strictly non-diagnostic; uses saved predicted_class without probability threshold inference.
    if predictions:
        try:
            latest_pred = predictions[0]
            pred_class = getattr(latest_pred, "predicted_class", None)

            pred_created = getattr(latest_pred, "created_at", None)
            pred_time_str = (
                pred_created.isoformat()
                if pred_created and hasattr(pred_created, "isoformat")
                else now_str
            )

            if pred_class == 2:
                notifications.append(Notification(
                    id=f"assessment_pred_{latest_pred.id}",
                    type="assessment",
                    category="assessment",
                    title="Assessment Result: Class 2 — Diabetes",
                    message="Your latest assessment classification is Class 2 — Diabetes. This is an educational research model estimate, NOT a medical diagnosis. Please consult a qualified healthcare professional for clinical diagnostic evaluation.",
                    action_label="View Insights",
                    action_url="/insights",
                    created_at=pred_time_str,
                    is_urgent=True,
                ))
            elif pred_class == 1:
                notifications.append(Notification(
                    id=f"assessment_pred_{latest_pred.id}",
                    type="assessment",
                    category="assessment",
                    title="Assessment Result: Class 1 — Prediabetes",
                    message="Your latest assessment classification is Class 1 — Prediabetes. Note that the research model has documented sensitivity limitations (Recall: 30.89%) for this class. Consult a physician for formal clinical testing.",
                    action_label="View Insights",
                    action_url="/insights",
                    created_at=pred_time_str,
                    is_urgent=False,
                ))
            elif pred_class == 0:
                notifications.append(Notification(
                    id=f"assessment_pred_{latest_pred.id}",
                    type="assessment",
                    category="assessment",
                    title="Assessment Result: Class 0 — No Diabetes",
                    message="Your latest assessment classification is Class 0 — No Diabetes. Continue maintaining healthy habits, balanced nutrition, and routine primary care check-ups.",
                    action_label="View Analytics",
                    action_url="/analytics",
                    created_at=pred_time_str,
                    is_urgent=False,
                ))
        except Exception as e:
            logger.debug("Could not generate assessment-triggered notification: %s", e)

    # 5. AI Assistant guidance (informational)
    notifications.append(Notification(
        id="healthbot_info",
        type="info",
        category="informational",
        title="Research Assistant (HealthBot)",
        message="Consult HealthBot for questions regarding CDC BRFSS health indicators, validation metrics, or model methodology.",
        action_label=None,
        action_url=None,
        created_at=now_str,
        is_urgent=False,
    ))

    # 6. Longitudinal analytics reminder (after 2+ assessments)
    if total_assessments >= 2:
        notifications.append(Notification(
            id="analytics_available",
            type="info",
            category="informational",
            title="Longitudinal Analytics Available",
            message=f"You have recorded {total_assessments} assessments. View longitudinal trends, indicator comparisons, and metric trajectories in Personal Analytics.",
            action_label="Open Analytics",
            action_url="/analytics",
            created_at=now_str,
            is_urgent=False,
        ))

    return notifications

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
