"""
backend/app/models/assessment.py
==================================
SQLAlchemy ORM model for the assessments table.
Stores all 21 CDC BRFSS 2015 health indicator features.
user_id is nullable to support anonymous / guest assessments.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, SmallInteger, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Optional link to a registered user (nullable = guest assessment)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── 21 CDC BRFSS 2015 health indicator features ─────────────────────────
    # Binary features (0/1)
    HighBP: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    HighChol: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    CholCheck: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    Smoker: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    Stroke: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    HeartDiseaseorAttack: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PhysActivity: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    Fruits: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    Veggies: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    HvyAlcoholConsump: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    AnyHealthcare: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    NoDocbcCost: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    DiffWalk: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    Sex: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # Continuous feature
    BMI: Mapped[float] = mapped_column(Float, nullable=False)

    # Count features (0–30)
    MentHlth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    PhysHlth: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # Ordinal features
    GenHlth: Mapped[int] = mapped_column(SmallInteger, nullable=False)   # 1–5
    Age: Mapped[int] = mapped_column(SmallInteger, nullable=False)       # 1–13
    Education: Mapped[int] = mapped_column(SmallInteger, nullable=False) # 1–6
    Income: Mapped[int] = mapped_column(SmallInteger, nullable=False)    # 1–8

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User | None"] = relationship("User", back_populates="assessments")  # noqa: F821
    prediction: Mapped["Prediction | None"] = relationship(  # noqa: F821
        "Prediction", back_populates="assessment", uselist=False,
        cascade="all, delete-orphan",
    )

    @property
    def user_name(self) -> str | None:
        return self.user.full_name if self.user else None

    def __repr__(self) -> str:
        return f"<Assessment id={self.id} user_id={self.user_id}>"

