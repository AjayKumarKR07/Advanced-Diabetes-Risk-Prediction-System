"""
backend/app/services/db_service.py
====================================
Database service layer: CRUD operations for assessments and predictions.
All writes are transactional; rollback is handled by the caller / get_db dependency.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.assessment import Assessment
from backend.app.models.prediction import Prediction
from backend.app.models.user import User
from backend.app.schemas.db_schemas import AssessmentCreate


# ---------------------------------------------------------------------------
# Assessment CRUD
# ---------------------------------------------------------------------------

def create_assessment(db: Session, data: AssessmentCreate) -> Assessment:
    """Persist a new assessment record and return it with its assigned id."""
    assessment = Assessment(
        user_id=data.user_id,
        HighBP=data.HighBP,
        HighChol=data.HighChol,
        CholCheck=data.CholCheck,
        BMI=data.BMI,
        Smoker=data.Smoker,
        Stroke=data.Stroke,
        HeartDiseaseorAttack=data.HeartDiseaseorAttack,
        PhysActivity=data.PhysActivity,
        Fruits=data.Fruits,
        Veggies=data.Veggies,
        HvyAlcoholConsump=data.HvyAlcoholConsump,
        AnyHealthcare=data.AnyHealthcare,
        NoDocbcCost=data.NoDocbcCost,
        GenHlth=data.GenHlth,
        MentHlth=data.MentHlth,
        PhysHlth=data.PhysHlth,
        DiffWalk=data.DiffWalk,
        Sex=data.Sex,
        Age=data.Age,
        Education=data.Education,
        Income=data.Income,
    )
    db.add(assessment)
    db.flush()   # populate assessment.id without committing
    return assessment


def get_assessment(db: Session, assessment_id: int) -> Optional[Assessment]:
    return db.get(Assessment, assessment_id)


def list_assessments(
    db: Session, page: int = 1, page_size: int = 20, user_id: Optional[int] = None
) -> tuple[int, list[Assessment]]:
    offset = (page - 1) * page_size
    stmt = select(Assessment)
    count_stmt = select(func.count()).select_from(Assessment)
    if user_id is not None:
        stmt = stmt.where(Assessment.user_id == user_id)
        count_stmt = count_stmt.where(Assessment.user_id == user_id)
    total = db.scalar(count_stmt) or 0
    items = list(
        db.scalars(
            stmt.order_by(Assessment.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
    )
    return total, items


def delete_assessment(db: Session, assessment_id: int) -> bool:
    """
    Delete an assessment and its cascaded prediction record.
    Returns True if record was found and deleted, False otherwise.
    """
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        return False
    db.delete(assessment)
    db.flush()
    return True


# ---------------------------------------------------------------------------
# Prediction CRUD
# ---------------------------------------------------------------------------

def create_prediction(
    db: Session,
    assessment_id: int,
    predicted_class: int,
    predicted_label: str,
    risk_tier: str,
    confidence: float,
    prob_no_diabetes: float,
    prob_prediabetes: float,
    prob_diabetes: float,
    model_version: str,
) -> Prediction:
    """Persist a prediction record linked to an assessment."""
    pred = Prediction(
        assessment_id=assessment_id,
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        risk_tier=risk_tier,
        confidence=confidence,
        probability_no_diabetes=prob_no_diabetes,
        probability_prediabetes=prob_prediabetes,
        probability_diabetes=prob_diabetes,
        model_version=model_version,
    )
    db.add(pred)
    db.flush()
    return pred


def get_prediction(db: Session, prediction_id: int) -> Optional[Prediction]:
    return db.get(Prediction, prediction_id)


def get_prediction_by_assessment(
    db: Session, assessment_id: int
) -> Optional[Prediction]:
    return db.scalar(
        select(Prediction).where(Prediction.assessment_id == assessment_id)
    )


def list_predictions(
    db: Session, page: int = 1, page_size: int = 20, user_id: Optional[int] = None
) -> tuple[int, list[Prediction]]:
    offset = (page - 1) * page_size
    stmt = select(Prediction)
    count_stmt = select(func.count()).select_from(Prediction)
    if user_id is not None:
        stmt = stmt.join(Prediction.assessment).where(Assessment.user_id == user_id)
        count_stmt = count_stmt.join(Prediction.assessment).where(Assessment.user_id == user_id)
    total = db.scalar(count_stmt) or 0
    items = list(
        db.scalars(
            stmt.order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
    )
    return total, items


# ---------------------------------------------------------------------------
# Dashboard aggregates
# ---------------------------------------------------------------------------

def get_dashboard_stats(db: Session, user_id: Optional[int] = None) -> dict:
    """Return aggregate statistics for the Dashboard page, optionally filtered by user."""
    assess_stmt = select(func.count()).select_from(Assessment)
    pred_stmt = select(func.count()).select_from(Prediction)
    dist_stmt = select(Prediction.predicted_class, func.count()).group_by(Prediction.predicted_class)
    recent_stmt = select(Assessment).order_by(Assessment.created_at.desc()).limit(5)

    if user_id is not None:
        assess_stmt = assess_stmt.where(Assessment.user_id == user_id)
        pred_stmt = pred_stmt.join(Prediction.assessment).where(Assessment.user_id == user_id)
        dist_stmt = (
            select(Prediction.predicted_class, func.count())
            .join(Prediction.assessment)
            .where(Assessment.user_id == user_id)
            .group_by(Prediction.predicted_class)
        )
        recent_stmt = (
            select(Assessment)
            .where(Assessment.user_id == user_id)
            .order_by(Assessment.created_at.desc())
            .limit(5)
        )

    total_assessments = db.scalar(assess_stmt) or 0
    total_predictions = db.scalar(pred_stmt) or 0

    # Class distribution
    rows = db.execute(dist_stmt).all()
    class_dist = {0: 0, 1: 0, 2: 0}
    for cls, cnt in rows:
        class_dist[cls] = cnt

    # Recent assessments
    recent = list(db.scalars(recent_stmt))

    return {
        "total_assessments": total_assessments,
        "total_predictions": total_predictions,
        "class_distribution": class_dist,
        "recent_assessments": recent,
    }


# ---------------------------------------------------------------------------
# User CRUD (Phase 6.9 Authentication)
# ---------------------------------------------------------------------------

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieve user by exact email match (case-insensitive)."""
    return db.scalar(
        select(User).where(func.lower(User.email) == email.strip().lower())
    )


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Retrieve user by primary key ID."""
    return db.get(User, user_id)


def create_user(
    db: Session,
    full_name: str,
    email: str,
    hashed_password: str,
) -> User:
    """Persist a new user with hashed password. Flushes to populate ID."""
    user = User(
        full_name=full_name.strip(),
        email=email.strip().lower(),
        hashed_password=hashed_password,
    )
    db.add(user)
    db.flush()
    return user


def update_user_profile(
    db: Session,
    user: User,
    full_name: str,
) -> User:
    """Update user's full name."""
    user.full_name = full_name.strip()
    db.flush()
    return user


def update_user_password(
    db: Session,
    user: User,
    hashed_password: str,
) -> User:
    """Update user's password hash."""
    user.hashed_password = hashed_password
    db.flush()
    return user



