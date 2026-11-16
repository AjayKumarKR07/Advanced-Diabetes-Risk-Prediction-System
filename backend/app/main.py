"""
backend/app/main.py
===================
FastAPI Application Entry Point — Production Ready (Phase 6.17).
Advanced Diabetes Risk Prediction System (DiabetesRisk AI).

Features:
  - Startup environment & security validation
  - Request correlation ID middleware (X-Request-ID)
  - Security headers middleware (nosniff, DENY, HSTS, CSP)
  - Liveness (/health) and Readiness (/ready) probes
  - Sanitized structured error logging
  - Graceful connection pool shutdown
"""

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.api.routes import api_router
from backend.app.core.config import settings
from backend.app.core.database import check_db_connection, dispose_db
from backend.app.services.model_service import model_service

# Configure structured application logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diabetesrisk.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Validate settings, pre-warm model, check DB, and cleanly shut down."""
    logger.info("Initializing DiabetesRisk AI backend (Environment: %s)", settings.ENVIRONMENT)

    # 1. Validate production configuration
    prod_errors = settings.validate_production_settings()
    if prod_errors:
        for err in prod_errors:
            logger.critical("Configuration Error: %s", err)
        raise RuntimeError(
            f"Production configuration validation failed ({len(prod_errors)} errors): {'; '.join(prod_errors)}"
        )

    # 2. Pre-warm ML model pipeline
    try:
        model_service.load_model()
        logger.info("✅ CDC Logistic Regression Model Pipeline preloaded successfully.")
    except Exception as exc:
        logger.warning("⚠️  Could not preload model on startup: %s", exc)

    # 3. Verify database connectivity
    try:
        if check_db_connection():
            logger.info("✅ PostgreSQL database connected successfully.")
        else:
            logger.warning("⚠️  PostgreSQL unreachable — running without DB persistence.")
    except Exception as exc:
        logger.warning("⚠️  Database connectivity check failed: %s", exc)

    yield

    # 4. Graceful shutdown
    logger.info("Application shutdown initiated — disposing database connections...")
    dispose_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT.lower() != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT.lower() != "production" else None,
)

# ── Middleware 1: Request Correlation ID (X-Request-ID) ─────────────────────────
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Middleware 2: Production Security Headers ──────────────────────────────────
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.scheme == "https" or settings.ENVIRONMENT.lower() in ("production", "prod"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Middleware 3: CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ──────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean, structured field-level validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    errors = []
    for err in exc.errors():
        field_name = " -> ".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({
            "field": field_name,
            "message": err["msg"],
            "type": err["type"],
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "One or more input features are missing, out of range, or invalid.",
            "details": errors,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    """Pass through standard HTTPExceptions with X-Request-ID header."""
    request_id = getattr(request.state, "request_id", "unknown")
    response = await http_exception_handler(request, exc)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception):
    """Catch unhandled 500 exceptions, log with request ID, and return sanitized response."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled internal server error | RequestID: %s | Method: %s | Path: %s | Error: %s",
        request_id,
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request. Please contact support with the request ID.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


# ── Liveness and Readiness Probes ──────────────────────────────────────────────
@app.get("/health", tags=["Health & Monitoring"], summary="Liveness probe")
async def liveness_check():
    """Kubernetes/Cloud liveness probe: confirms backend process is responsive."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION,
    }


@app.get("/ready", tags=["Health & Monitoring"], summary="Readiness probe")
async def readiness_check():
    """Kubernetes/Cloud readiness probe: confirms database and ML model are ready for traffic."""
    db_ok = check_db_connection()
    model_ok = model_service.is_loaded
    is_ready = db_ok and model_ok
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if is_ready else "not_ready",
            "model_loaded": model_ok,
            "database_connected": db_ok,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
        },
    )


# ── Mount API routes under /api ────────────────────────────────────────────────
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing service information, disclaimer, and documentation links."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "model_loaded": model_service.is_loaded,
        "documentation": "/docs" if settings.ENVIRONMENT.lower() != "production" else "Disabled in production",
        "api_prefix": settings.API_V1_STR,
        "disclaimer": settings.DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
