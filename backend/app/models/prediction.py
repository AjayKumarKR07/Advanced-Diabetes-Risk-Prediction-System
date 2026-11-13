"""
backend/app/models/prediction.py
==================================
SQLAlchemy ORM model for the predictions table.
Stores ML inference outputs linked to an assessment record.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, SmallInteger, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    assessment_id: Mapped[int] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"),
        unique=True,   # one prediction per assessment
        nullable=False,
        index=True,
    )

    # Inference outputs
    predicted_class: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Three class probabilities (model-estimated, not clinically calibrated)
    probability_no_diabetes: Mapped[float] = mapped_column(Float, nullable=False)
    probability_prediabetes: Mapped[float] = mapped_column(Float, nullable=False)
    probability_diabetes: Mapped[float] = mapped_column(Float, nullable=False)

    model_version: Mapped[str] = mapped_column(
        String(128), nullable=False, default="LogisticRegression-GroupAware"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to assessment
    assessment: Mapped["Assessment"] = relationship(  # noqa: F821
        "Assessment", back_populates="prediction"
    )

    @property
    def user_id(self) -> int | None:
        return self.assessment.user_id if self.assessment else None

    @property
    def user_name(self) -> str | None:
        if self.assessment and self.assessment.user:
            return self.assessment.user.full_name
        return None

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} assessment_id={self.assessment_id} "
            f"class={self.predicted_class}>"
        )

