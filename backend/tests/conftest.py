"""
backend/tests/conftest.py
=========================
Shared pytest configuration for the FastAPI backend test suite.
Pre-loads the model service singleton once per session.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def preload_model():
    """
    Session-scoped fixture: loads the DiabetesModelService singleton before
    any test accesses it, so standalone unit tests can safely inspect
    model_service.is_loaded without triggering a lazy load race.
    """
    from backend.app.services.model_service import model_service

    if not model_service.is_loaded:
        model_service.load_model()
    yield model_service
