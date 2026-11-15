"""
alembic/env.py
==============
Alembic environment configuration.
DATABASE_URL is sourced from backend/.env via pydantic-settings (never hardcoded).
All ORM models are imported here so their metadata is registered on Base.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so backend.* imports resolve
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Import settings (reads backend/.env) and declarative Base
# ---------------------------------------------------------------------------
from backend.app.core.config import settings  # noqa: E402
from backend.app.core.database import Base  # noqa: E402

# Import all ORM models so their tables are registered on Base.metadata
from backend.app.models import user, assessment, prediction, user_settings, revoked_token  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Build a SQLAlchemy engine directly from settings (bypasses ConfigParser
# interpolation issues with special chars like '%' in passwords).
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine, pool  # noqa: E402


def _make_engine(url: str):
    return create_engine(url, poolclass=pool.NullPool)


# ---------------------------------------------------------------------------
# Run migrations
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout/file)."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to live DB)."""
    connectable = _make_engine(settings.DATABASE_URL)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
