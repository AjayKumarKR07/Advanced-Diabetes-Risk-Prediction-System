"""
backend/app/schemas/analytics_schemas.py
========================================
Pydantic schemas for personal health analytics, trend analysis,
indicator comparison, and dashboard reporting (Phase 6.15).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BmiTrendPoint(BaseModel):
    assessment_id: int
    date: str
    full_date: str
    timestamp: datetime
    bmi: float
    category: str


class BmiTrendSummary(BaseModel):
    latest: Optional[float] = None
    previous: Optional[float] = None
    change: Optional[float] = None
    category: Optional[str] = None
    count: int = 0


class BmiTrendsResponse(BaseModel):
    points: List[BmiTrendPoint]
    summary: BmiTrendSummary


class HealthTrendPoint(BaseModel):
    assessment_id: int
    date: str
    full_date: str
    timestamp: datetime
    gen_hlth: int
    gen_hlth_label: str
    phys_hlth: int
    ment_hlth: int


class HealthTrendSummary(BaseModel):
    latest: Optional[int] = None
    latest_label: Optional[str] = None
    previous: Optional[int] = None
    change: Optional[int] = None
    count: int = 0


class HealthTrendsResponse(BaseModel):
    points: List[HealthTrendPoint]
    summary: HealthTrendSummary


class PredictionDistributionResponse(BaseModel):
    total: int
    counts: Dict[str, int] = Field(
        default_factory=lambda: {"0": 0, "1": 0, "2": 0},
        description="Prediction counts per class: 0 (No Diabetes), 1 (Prediabetes), 2 (Diabetes)",
    )
    percentages: Dict[str, float] = Field(
        default_factory=lambda: {"0": 0.0, "1": 0.0, "2": 0.0},
        description="Percentage of predictions per class (0-100)",
    )


class IndicatorComparisonRecord(BaseModel):
    assessment_id: int
    date: str
    full_date: str
    timestamp: datetime
    bmi: float
    bmi_category: str
    high_bp: int
    high_chol: int
    phys_activity: int
    smoker: int
    gen_hlth: int
    gen_hlth_label: str
    phys_hlth: int
    ment_hlth: int
    diff_walk: int
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None


class IndicatorComparisonResponse(BaseModel):
    total_records: int
    records: List[IndicatorComparisonRecord]


class TimelineItem(BaseModel):
    assessment_id: int
    prediction_id: Optional[int] = None
    created_at: datetime
    date: str
    full_date: str
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    probability_no_diabetes: Optional[float] = None
    probability_prediabetes: Optional[float] = None
    probability_diabetes: Optional[float] = None
    bmi: float
    bmi_category: str
    gen_hlth: int
    gen_hlth_label: str
    user_name: Optional[str] = None


class TimelineResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[TimelineItem]


class AnalyticsOverviewResponse(BaseModel):
    total_assessments: int
    assessments_this_month: int
    latest_assessment_id: Optional[int] = None
    latest_prediction_id: Optional[int] = None
    latest_assessment_date: Optional[str] = None
    latest_predicted_class: Optional[int] = None
    latest_predicted_label: Optional[str] = None
    latest_confidence: Optional[float] = None
    latest_probabilities: Optional[Dict[str, float]] = None
    latest_bmi: Optional[float] = None
    latest_bmi_category: Optional[str] = None
    latest_gen_hlth: Optional[int] = None
    latest_gen_hlth_label: Optional[str] = None
    bmi_summary: BmiTrendSummary
    gen_hlth_summary: HealthTrendSummary
    class_distribution: Dict[str, int]
    filter_applied: Dict[str, Optional[str]]

