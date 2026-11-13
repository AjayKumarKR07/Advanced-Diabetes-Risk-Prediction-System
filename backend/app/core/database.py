"""
backend/app/core/database.py
=============================
SQLAlchemy 2.x engine, session factory, declarative base and
FastAPI dependency for database session injection.

Connection is driven entirely by DATABASE_URL in backend/.env.
No credentials are hardcoded.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,          # validates connection liveness before use
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,           # recycle connections every 30 min
    echo=False,                  # set True for SQL debug output
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative base (all ORM models inherit from this)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a scoped DB session per request
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    Provide a transactional database session for the duration of a request.
    Rolls back on any unhandled exception; always closes on exit.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Context-manager wrapper (useful outside FastAPI request scope)
# ---------------------------------------------------------------------------

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Synchronous context-manager version of get_db for scripts and tests."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

def check_db_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as exc:
        logger.error("Database connection failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Schema initialisation (create_all) — used in tests and dev
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables that do not yet exist. Safe to call repeatedly."""
    # Import models so their metadata is registered on Base
    from backend.app.models import assessment, prediction, user  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")

