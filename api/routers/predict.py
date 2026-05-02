from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from api.core.config import get_settings
from api.core.logging import get_logger
from api.domain.manifest import ModelManifest
from api.domain.registry import ModelRegistry
from api.schemas.common import error_response, success_response
from api.schemas.predict import (
    BatchPredictResponse,
    PredictResponse,
    build_batch_predict_request_validator,
    build_predict_request_validator,
)
from api.services.predictor import predict_batch, predict_one

logger = get_logger("router.predict")
router = APIRouter(prefix="/models", tags=["predict"])

_validator_cache_single: dict[tuple[str, str], type] = {}
_validator_cache_batch: dict[tuple[str, str], type] = {}


def _get_validator_single(manifest: ModelManifest) -> type:
    key = (manifest.model_id, str(manifest.model_path))
    cached = _validator_cache_single.get(key)
    if cached is None:
        cached = build_predict_request_validator(manifest)
        _validator_cache_single[key] = cached
    return cached


def _get_validator_batch(manifest: ModelManifest) -> type:
    key = (manifest.model_id, str(manifest.model_path))
    cached = _validator_cache_batch.get(key)
    if cached is None:
        cached = build_batch_predict_request_validator(manifest)
        _validator_cache_batch[key] = cached
    return cached


def _require_manifest(registry: ModelRegistry, target: str, algorithm: str) -> ModelManifest:
    manifest = registry.get_manifest(target, algorithm)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: target='{target}', algorithm='{algorithm}'",
        )
    return manifest


@router.post("/{target}/{algorithm}/predict")
async def predict(target: str, algorithm: str, request: Request) -> dict:
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)

    body = await request.json()
    Validator = _get_validator_single(manifest)
    try:
        validated = Validator.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        field_path = ".".join(str(p) for p in first.get("loc", ()))
        return error_response(
            code="invalid_input",
            message=first.get("msg", "Input inválido"),
            field=field_path or None,
            model_id=manifest.model_id,
        )

    features_dict = validated.features.model_dump()
    result = predict_one(features_dict, manifest, registry)

    response = PredictResponse(
        predicted_class=result.predicted_class,
        probability=result.probability,
        threshold=result.threshold,
        risk_level=result.risk_level,
        calibrated=result.calibrated,
        prevalence_train=manifest.prevalence.get("train"),
        warnings=result.warnings,
    )
    return success_response(response.model_dump(mode="json"), model_id=manifest.model_id)


@router.post("/{target}/{algorithm}/predict/batch")
async def predict_batch_endpoint(target: str, algorithm: str, request: Request) -> dict:
    settings = get_settings()
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)

    body = await request.json()
    Validator = _get_validator_batch(manifest)
    try:
        validated = Validator.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        field_path = ".".join(str(p) for p in first.get("loc", ()))
        return error_response(
            code="invalid_input",
            message=first.get("msg", "Input inválido"),
            field=field_path or None,
            model_id=manifest.model_id,
        )

    items = validated.items
    if len(items) == 0:
        return error_response(
            code="empty_batch",
            message="Batch vacío.",
            model_id=manifest.model_id,
        )
    if len(items) > settings.max_batch_size:
        return error_response(
            code="batch_too_large",
            message=f"Batch excede el máximo permitido ({settings.max_batch_size}).",
            model_id=manifest.model_id,
        )

    features_list = [item.model_dump() for item in items]
    results = predict_batch(features_list, manifest, registry)

    payload = BatchPredictResponse(
        predictions=[
            PredictResponse(
                predicted_class=r.predicted_class,
                probability=r.probability,
                threshold=r.threshold,
                risk_level=r.risk_level,
                calibrated=r.calibrated,
                prevalence_train=manifest.prevalence.get("train"),
                warnings=r.warnings,
            )
            for r in results
        ],
        n=len(results),
    )
    return success_response(payload.model_dump(mode="json"), model_id=manifest.model_id)
