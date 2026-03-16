from fastapi import APIRouter

from api.core.config import settings
from api.schemas.responses import HealthResponse
from api.services.model_registry import registry

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Returns API status and number of loaded models."""
    return HealthResponse(
        status="ok" if len(registry) > 0 else "degraded",
        models_loaded=len(registry),
        version=settings.app_version,
        shap_enabled=settings.enable_shap,
    )
