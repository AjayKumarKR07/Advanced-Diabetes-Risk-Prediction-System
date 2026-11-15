"""
backend/app/services/analytics_service.py
=========================================
SQLAlchemy query service for personal health analytics, longitudinal BMI and
health indicator trends, prediction distributions, and assessment timelines.
All operations are strictly scoped to the authenticated user ID.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from backend.app.models.assessment import Assessment
from backend.app.models.prediction import Prediction
from backend.app.schemas.analytics_schemas import (
    AnalyticsOverviewResponse,
    BmiTrendPoint,
    BmiTrendsResponse,
    BmiTrendSummary,
    HealthTrendPoint,
    HealthTrendsResponse,
    HealthTrendSummary,
    IndicatorComparisonRecord,
    IndicatorComparisonResponse,
    PredictionDistributionResponse,
    TimelineItem,
    TimelineResponse,
)

GEN_HLTH_LABELS: Dict[int, str] = {
    1: "1 — Excellent",
    2: "2 — Very Good",
    3: "3 — Good",
    4: "4 — Fair",
    5: "5 — Poor",
}


def get_bmi_category(bmi: float) -> str:
    """Return descriptive screening category for adult BMI according to WHO/CDC."""
    if bmi < 18.5:
        return "Underweight (<18.5)"
    if bmi < 25.0:
        return "Normal range (18.5–24.9)"
    if bmi < 30.0:
        return "Overweight (25.0–29.9)"
    return "Obesity range (≥30.0)"


def parse_date_filter(
    start_date: Optional[str], end_date: Optional[str]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse optional ISO date/timestamp strings into timezone-aware UTC datetimes.
    Raises HTTPException(400) if date format is invalid or start_date > end_date.
    """
    dt_start = None
    dt_end = None

    if start_date and start_date.strip():
        s = start_date.strip().replace("Z", "+00:00")
        try:
            dt_start = datetime.fromisoformat(s)
            if dt_start.tzinfo is None:
                dt_start = dt_start.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid start_date format: '{start_date}'. Expected ISO format (e.g. YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
            )

    if end_date and end_date.strip():
        e = end_date.strip().replace("Z", "+00:00")
        try:
            dt_end = datetime.fromisoformat(e)
            if dt_end.tzinfo is None:
                dt_end = dt_end.replace(tzinfo=timezone.utc)
            if len(e) <= 10:
                dt_end = dt_end.replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid end_date format: '{end_date}'. Expected ISO format (e.g. YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
            )

    if dt_start is not None and dt_end is not None:
        if dt_start > dt_end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date range: start_date ({start_date}) cannot be after end_date ({end_date}).",
            )

    return dt_start, dt_end


def _apply_date_filter(stmt, dt_start: Optional[datetime], dt_end: Optional[datetime]):
    """Apply start and end timestamp filters to an Assessment-based query."""
    if dt_start is not None:
        stmt = stmt.where(Assessment.created_at >= dt_start)
    if dt_end is not None:
        stmt = stmt.where(Assessment.created_at <= dt_end)
    return stmt


def get_analytics_overview(
    db: Session,
    user_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> AnalyticsOverviewResponse:
    """
    Compute comprehensive KPI overview for the authenticated user,
    optionally restricted to the given date window.
    """
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    # Base query for user's assessments with eager loaded predictions
    stmt = (
        select(Assessment)
        .options(joinedload(Assessment.prediction))
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
    )
    stmt = _apply_date_filter(stmt, dt_start, dt_end)
    assessments: List[Assessment] = list(db.scalars(stmt).unique())

    total_assessments = len(assessments)

    # Assessments completed this calendar month (UTC)
    now_utc = datetime.now(timezone.utc)
    month_start = datetime(now_utc.year, now_utc.month, 1, tzinfo=timezone.utc)
    assessments_this_month = sum(
        1 for a in assessments if a.created_at and a.created_at >= month_start
    )

    # Latest assessment and prediction
    latest_assessment = assessments[0] if assessments else None
    latest_pred = latest_assessment.prediction if latest_assessment else None

    # Latest metadata
    latest_assessment_id = latest_assessment.id if latest_assessment else None
    latest_prediction_id = latest_pred.id if latest_pred else None
    latest_assessment_date = (
        latest_assessment.created_at.strftime("%b %d, %Y")
        if latest_assessment and latest_assessment.created_at
        else None
    )

    latest_predicted_class = latest_pred.predicted_class if latest_pred else None
    latest_predicted_label = latest_pred.predicted_label if latest_pred else None
    latest_confidence = latest_pred.confidence if latest_pred else None
    latest_probabilities = None
    if latest_pred:
        latest_probabilities = {
            "no_diabetes": round(latest_pred.probability_no_diabetes, 4),
            "prediabetes": round(latest_pred.probability_prediabetes, 4),
            "diabetes": round(latest_pred.probability_diabetes, 4),
        }

    latest_bmi = round(latest_assessment.BMI, 1) if latest_assessment else None
    latest_bmi_category = get_bmi_category(latest_assessment.BMI) if latest_assessment else None
    latest_gen_hlth = latest_assessment.GenHlth if latest_assessment else None
    latest_gen_hlth_label = (
        GEN_HLTH_LABELS.get(latest_assessment.GenHlth, f"Rating {latest_assessment.GenHlth}")
        if latest_assessment
        else None
    )

    # BMI Trend Summary (chronological: assessments reversed from desc to asc)
    chrono_assessments = list(reversed(assessments))
    bmi_summary = BmiTrendSummary(count=total_assessments)
    if total_assessments >= 1:
        bmi_summary.latest = round(latest_assessment.BMI, 1)
        bmi_summary.category = get_bmi_category(latest_assessment.BMI)
    if total_assessments >= 2:
        prev_a = assessments[1]
        bmi_summary.previous = round(prev_a.BMI, 1)
        bmi_summary.change = round(latest_assessment.BMI - prev_a.BMI, 1)

    # General Health Summary
    gen_hlth_summary = HealthTrendSummary(count=total_assessments)
    if total_assessments >= 1:
        gen_hlth_summary.latest = latest_assessment.GenHlth
        gen_hlth_summary.latest_label = GEN_HLTH_LABELS.get(
            latest_assessment.GenHlth, str(latest_assessment.GenHlth)
        )
    if total_assessments >= 2:
        prev_a = assessments[1]
        gen_hlth_summary.previous = prev_a.GenHlth
        gen_hlth_summary.change = latest_assessment.GenHlth - prev_a.GenHlth

    # Class distribution
    class_dist: Dict[str, int] = {"0": 0, "1": 0, "2": 0}
    for a in assessments:
        if a.prediction:
            cls_key = str(a.prediction.predicted_class)
            class_dist[cls_key] = class_dist.get(cls_key, 0) + 1

    return AnalyticsOverviewResponse(
        total_assessments=total_assessments,
        assessments_this_month=assessments_this_month,
        latest_assessment_id=latest_assessment_id,
        latest_prediction_id=latest_prediction_id,
        latest_assessment_date=latest_assessment_date,
        latest_predicted_class=latest_predicted_class,
        latest_predicted_label=latest_predicted_label,
        latest_confidence=latest_confidence,
        latest_probabilities=latest_probabilities,
        latest_bmi=latest_bmi,
        latest_bmi_category=latest_bmi_category,
        latest_gen_hlth=latest_gen_hlth,
        latest_gen_hlth_label=latest_gen_hlth_label,
        bmi_summary=bmi_summary,
        gen_hlth_summary=gen_hlth_summary,
        class_distribution=class_dist,
        filter_applied={"start_date": start_date, "end_date": end_date},
    )


def get_prediction_distribution(
    db: Session,
    user_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> PredictionDistributionResponse:
    """Retrieve historical prediction outcome counts and percentages for user."""
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    stmt = (
        select(Prediction.predicted_class, func.count())
        .join(Prediction.assessment)
        .where(Assessment.user_id == user_id)
    )
    if dt_start is not None:
        stmt = stmt.where(Assessment.created_at >= dt_start)
    if dt_end is not None:
        stmt = stmt.where(Assessment.created_at <= dt_end)

    stmt = stmt.group_by(Prediction.predicted_class)
    rows = db.execute(stmt).all()

    counts: Dict[str, int] = {"0": 0, "1": 0, "2": 0}
    total = 0
    for cls, cnt in rows:
        key = str(cls)
        counts[key] = cnt
        total += cnt

    percentages: Dict[str, float] = {"0": 0.0, "1": 0.0, "2": 0.0}
    if total > 0:
        for k, v in counts.items():
            percentages[k] = round((v / total) * 100.0, 1)

    return PredictionDistributionResponse(
        total=total,
        counts=counts,
        percentages=percentages,
    )


def get_bmi_trends(
    db: Session,
    user_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BmiTrendsResponse:
    """
    Retrieve chronological BMI trend points for user with reference thresholds.
    Points are ordered oldest to newest.
    """
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    stmt = (
        select(Assessment)
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.asc())
    )
    stmt = _apply_date_filter(stmt, dt_start, dt_end)
    records = list(db.scalars(stmt))

    points: List[BmiTrendPoint] = []
    for r in records:
        if r.created_at is None:
            continue
        bmi_val = round(float(r.BMI), 1)
        points.append(
            BmiTrendPoint(
                assessment_id=r.id,
                date=r.created_at.strftime("%b %d"),
                full_date=r.created_at.strftime("%b %d, %Y %I:%M %p"),
                timestamp=r.created_at,
                bmi=bmi_val,
                category=get_bmi_category(bmi_val),
            )
        )

    summary = BmiTrendSummary(count=len(points))
    if len(points) >= 1:
        summary.latest = points[-1].bmi
        summary.category = points[-1].category
    if len(points) >= 2:
        summary.previous = points[-2].bmi
        summary.change = round(points[-1].bmi - points[-2].bmi, 1)

    return BmiTrendsResponse(points=points, summary=summary)


def get_health_trends(
    db: Session,
    user_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> HealthTrendsResponse:
    """
    Retrieve chronological self-reported General Health, PhysHlth, and MentHlth
    survey ratings for user.
    """
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    stmt = (
        select(Assessment)
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.asc())
    )
    stmt = _apply_date_filter(stmt, dt_start, dt_end)
    records = list(db.scalars(stmt))

    points: List[HealthTrendPoint] = []
    for r in records:
        if r.created_at is None:
            continue
        points.append(
            HealthTrendPoint(
                assessment_id=r.id,
                date=r.created_at.strftime("%b %d"),
                full_date=r.created_at.strftime("%b %d, %Y %I:%M %p"),
                timestamp=r.created_at,
                gen_hlth=r.GenHlth,
                gen_hlth_label=GEN_HLTH_LABELS.get(r.GenHlth, f"Rating {r.GenHlth}"),
                phys_hlth=r.PhysHlth,
                ment_hlth=r.MentHlth,
            )
        )

    summary = HealthTrendSummary(count=len(points))
    if len(points) >= 1:
        summary.latest = points[-1].gen_hlth
        summary.latest_label = points[-1].gen_hlth_label
    if len(points) >= 2:
        summary.previous = points[-2].gen_hlth
        summary.change = points[-1].gen_hlth - points[-2].gen_hlth

    return HealthTrendsResponse(points=points, summary=summary)


def get_indicator_comparison(
    db: Session,
    user_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> IndicatorComparisonResponse:
    """
    Retrieve historical submitted values for key indicators across user assessments,
    eagerly joining prediction outcomes for complete longitudinal comparison.
    """
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    stmt = (
        select(Assessment)
        .options(joinedload(Assessment.prediction))
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
    )
    stmt = _apply_date_filter(stmt, dt_start, dt_end)
    records = list(db.scalars(stmt).unique())

    items: List[IndicatorComparisonRecord] = []
    for r in records:
        if r.created_at is None:
            continue
        bmi_val = round(float(r.BMI), 1)
        pred = r.prediction
        items.append(
            IndicatorComparisonRecord(
                assessment_id=r.id,
                date=r.created_at.strftime("%b %d, %Y"),
                full_date=r.created_at.strftime("%b %d, %Y %I:%M %p"),
                timestamp=r.created_at,
                bmi=bmi_val,
                bmi_category=get_bmi_category(bmi_val),
                high_bp=r.HighBP,
                high_chol=r.HighChol,
                phys_activity=r.PhysActivity,
                smoker=r.Smoker,
                gen_hlth=r.GenHlth,
                gen_hlth_label=GEN_HLTH_LABELS.get(r.GenHlth, f"Rating {r.GenHlth}"),
                phys_hlth=r.PhysHlth,
                ment_hlth=r.MentHlth,
                diff_walk=r.DiffWalk,
                predicted_class=pred.predicted_class if pred else None,
                predicted_label=pred.predicted_label if pred else None,
                confidence=round(pred.confidence, 4) if pred else None,
            )
        )

    return IndicatorComparisonResponse(total_records=len(items), records=items)


def get_assessment_timeline(
    db: Session,
    user_id: int,
    page: int = 1,
    page_size: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> TimelineResponse:
    """
    Retrieve paginated timeline of completed assessments joined with predictions,
    ordered newest to oldest.
    """
    dt_start, dt_end = parse_date_filter(start_date, end_date)

    count_stmt = select(func.count()).select_from(Assessment).where(Assessment.user_id == user_id)
    count_stmt = _apply_date_filter(count_stmt, dt_start, dt_end)
    total = db.scalar(count_stmt) or 0

    offset = (page - 1) * page_size
    stmt = (
        select(Assessment)
        .options(joinedload(Assessment.prediction), joinedload(Assessment.user))
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    stmt = _apply_date_filter(stmt, dt_start, dt_end)
    records = list(db.scalars(stmt).unique())

    items: List[TimelineItem] = []
    for r in records:
        if r.created_at is None:
            continue
        pred = r.prediction
        bmi_val = round(float(r.BMI), 1)
        items.append(
            TimelineItem(
                assessment_id=r.id,
                prediction_id=pred.id if pred else None,
                created_at=r.created_at,
                date=r.created_at.strftime("%b %d, %Y"),
                full_date=r.created_at.strftime("%b %d, %Y %I:%M %p"),
                predicted_class=pred.predicted_class if pred else None,
                predicted_label=pred.predicted_label if pred else None,
                confidence=round(pred.confidence, 4) if pred else None,
                probability_no_diabetes=round(pred.probability_no_diabetes, 4) if pred else None,
                probability_prediabetes=round(pred.probability_prediabetes, 4) if pred else None,
                probability_diabetes=round(pred.probability_diabetes, 4) if pred else None,
                bmi=bmi_val,
                bmi_category=get_bmi_category(bmi_val),
                gen_hlth=r.GenHlth,
                gen_hlth_label=GEN_HLTH_LABELS.get(r.GenHlth, f"Rating {r.GenHlth}"),
                user_name=r.user.full_name if r.user else None,
            )
        )

    return TimelineResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )

