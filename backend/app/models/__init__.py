"""
backend/app/models/__init__.py
================================
ORM model package — importing here ensures SQLAlchemy registers
all table metadata on Base before create_all() is called.
"""

from backend.app.models.user import User
from backend.app.models.assessment import Assessment
from backend.app.models.prediction import Prediction

__all__ = ["User", "Assessment", "Prediction"]

