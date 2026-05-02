from fastapi import APIRouter, Request

from api.schemas.common import success_response

router = APIRouter(prefix="/targets", tags=["targets"])


@router.get("")
async def list_targets(request: Request) -> dict:
    """Lista los target slugs servidos. Útil para que el frontend renderice el primer dropdown."""
    registry = request.app.state.registry
    return success_response(registry.list_targets())
