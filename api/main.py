"""
Preanesthesia Screening API
===========================
Generic REST API for uploading and serving preanesthesia screening ML models.
Models are stored as .joblib artifacts in the models/ directory and served
through a unified prediction interface that adapts to each model's feature set.

Startup
-------
    uvicorn api.main:app --reload               # development
    uvicorn api.main:app --host 0.0.0.0 --port 8000  # production

Interactive docs
----------------
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

# Force UTF-8 stdout on Windows so Spanish characters and symbols log correctly.
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.routers import health_router, models_router, predict_router
from api.services.model_registry import load_registry

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Preanesthesia Screening API starting ===")
    logger.info("Models directory: %s", settings.models_dir.resolve())
    load_registry()
    yield
    logger.info("=== API shutting down ===")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.description,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(models_router)
app.include_router(predict_router)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"], summary="API info")
def root():
    return JSONResponse(
        {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
        }
    )
