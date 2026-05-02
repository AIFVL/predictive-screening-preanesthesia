from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import get_settings
from api.core.logging import configure_logging, get_logger
from api.domain.registry import ModelRegistry
from api.routers import health, models, predict, targets

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(f"Booting API — models_dir={settings.models_dir}")

    registry = ModelRegistry(settings)
    registry.discover()
    app.state.settings = settings
    app.state.registry = registry

    if registry.n_registered() == 0:
        logger.warning("La API arrancó sin modelos registrados.")
    yield
    logger.info("API shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "API de inferencia para modelos de screening preanestésico. "
            "Lista modelos servidos (`/models`), expone su schema de features "
            "(`/models/{target}/{algorithm}/schema`) y produce predicciones "
            "calibradas (`/models/{target}/{algorithm}/predict`)."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(targets.router)
    app.include_router(models.router)
    app.include_router(predict.router)

    return app


app = create_app()
