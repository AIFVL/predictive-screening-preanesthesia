"""
Model management endpoints
==========================
GET  /models                 - list all loaded models
GET  /models/{model_id}      - model details + metrics + top features
POST /models/upload          - upload a new .joblib artifact
DELETE /models/{model_id}    - remove a model from memory (and disk)
"""

import io
import logging

import joblib
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.schemas.responses import (
    ModelDeleteResponse,
    ModelInfo,
    ModelListResponse,
    ModelUploadResponse,
)
from api.services.model_registry import registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelListResponse, summary="List available models")
def list_models():
    """Return metadata for every loaded model artifact."""
    infos = registry.list_infos()
    return ModelListResponse(models=infos, total=len(infos))


@router.get("/{model_id}", response_model=ModelInfo, summary="Get model details")
def get_model(model_id: str):
    """Return detailed metadata (metrics, top features, threshold) for a specific model."""
    try:
        return registry.get_info(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/upload",
    response_model=ModelUploadResponse,
    summary="Upload a new model artifact",
    status_code=201,
)
async def upload_model(
    file: UploadFile = File(..., description="Joblib artifact produced by scripts/train_and_save.py"),
    model_id: str = Query(
        ...,
        description="Unique identifier for this model (letters, digits, underscores only). "
        "If a model with this ID already exists it will be replaced.",
    ),
    overwrite: bool = Query(False, description="Allow overwriting an existing model ID."),
):
    """
    Upload a `.joblib` model artifact and register it for immediate use.

    The artifact must follow the schema documented in `api/services/model_registry.py`.
    """
    if not model_id.replace("_", "").isalnum():
        raise HTTPException(
            status_code=422,
            detail="model_id must contain only letters, digits and underscores.",
        )

    if model_id in registry.list_ids() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' already exists. Pass overwrite=true to replace it.",
        )

    if not file.filename.endswith(".joblib"):
        raise HTTPException(status_code=422, detail="File must have a .joblib extension.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        artifact = joblib.load(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not deserialise joblib file: {exc}")

    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=422,
            detail="Artifact must be a Python dict. See model_registry.py for the expected schema.",
        )

    try:
        info = registry.register(model_id, artifact, save_dir=settings.models_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    logger.info("Uploaded model '%s' (%s / %s).", model_id, info.model_name, info.version)
    return ModelUploadResponse(
        model_id=model_id,
        message=f"Model '{model_id}' registered successfully.",
        model_info=info,
    )


@router.delete(
    "/{model_id}",
    response_model=ModelDeleteResponse,
    summary="Remove a model",
)
def delete_model(
    model_id: str,
    delete_file: bool = Query(
        False, description="Also delete the .joblib file from disk."
    ),
):
    """Unload a model from memory and optionally delete its artifact file."""
    try:
        registry.remove(
            model_id,
            models_dir=settings.models_dir if delete_file else None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return ModelDeleteResponse(
        model_id=model_id,
        message=f"Model '{model_id}' removed"
        + (" and file deleted." if delete_file else " from memory."),
    )
