"""
backend/app/models/user_settings.py
===================================
SQLAlchemy ORM model for the user_settings table.
Stores theme and notification preferences per authenticated user.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class UserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    theme: Mapped[str] = mapped_column(String(32), default="system", nullable=False)
    notify_assessment_ready: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_security_alerts: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notify_product_updates: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="settings")  # noqa: F821

    def __repr__(self) -> str:
        return f"<UserSettings user_id={self.user_id} theme={self.theme!r}>"

