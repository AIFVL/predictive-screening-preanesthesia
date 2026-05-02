from fastapi import APIRouter, Request

from api.schemas.common import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe — la app responde."""
    return success_response({"status": "ok"})


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness probe — hay al menos un modelo registrado."""
    registry = request.app.state.registry
    n = registry.n_registered()
    return success_response({
        "status": "ready" if n > 0 else "no_models",
        "n_models": n,
        "cache_loaded": registry.cache_size(),
    })
