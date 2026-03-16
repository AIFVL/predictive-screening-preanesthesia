from .health import router as health_router
from .models import router as models_router
from .predict import router as predict_router

__all__ = ["health_router", "models_router", "predict_router"]
