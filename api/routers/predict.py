from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from api.core.logging import get_logger
from api.domain.manifest import ModelManifest
from api.domain.registry import ModelRegistry
from api.schemas.common import error_response, success_response
from api.schemas.predict import (
    ExplainResponse,
    PredictResponse,
    ShapContributionSchema,
)
from api.services.clinical_preprocessor import preprocess_raw
from api.services.explainer import explain_one
from api.services.predictor import predict_one

logger = get_logger("router.predict")
router = APIRouter(prefix="/models", tags=["predict"])


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
    """
    Acepta datos clínicos crudos (claves = nombre de columna del dataset),
    ejecuta el pipeline completo de limpieza y enriquecimiento, y devuelve
    la predicción calibrada.
    """
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)

    body = await request.json()
    patient = body.get("patient")
    if not isinstance(patient, dict):
        return error_response(
            code="invalid_input",
            message="El campo 'patient' es requerido y debe ser un objeto JSON.",
            model_id=manifest.model_id,
        )

    cleaning_cfg: dict = request.app.state.cleaning_cfg
    cache_dir: Path = request.app.state.settings.cache_dir

    try:
        df = preprocess_raw(
            patient=patient,
            manifest=manifest,
            cache_dir=cache_dir,
            cleaning_cfg=cleaning_cfg,
        )
    except Exception as exc:
        logger.exception(f"Error en preprocesamiento para {target}/{algorithm}: {exc}")
        return error_response(
            code="preprocessing_error",
            message=f"Error al preprocesar los datos clínicos: {exc}",
            model_id=manifest.model_id,
        )

    features_dict = df.iloc[0].to_dict()
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


@router.post("/{target}/{algorithm}/explain")
async def explain(target: str, algorithm: str, request: Request) -> dict:
    """
    Devuelve las top-N contribuciones SHAP para una observación clínica cruda.
    Body: { "patient": {...}, "top_n": 10 }
    """
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)

    body = await request.json()
    patient = body.get("patient")
    if not isinstance(patient, dict):
        return error_response(
            code="invalid_input",
            message="El campo 'patient' es requerido y debe ser un objeto JSON.",
            model_id=manifest.model_id,
        )

    top_n = int(body.get("top_n", 10))
    top_n = max(1, min(top_n, len(manifest.feature_names)))

    cleaning_cfg: dict = request.app.state.cleaning_cfg
    cache_dir: Path = request.app.state.settings.cache_dir

    try:
        contributions = explain_one(
            patient=patient,
            manifest=manifest,
            registry=registry,
            cache_dir=cache_dir,
            cleaning_cfg=cleaning_cfg,
            top_n=top_n,
        )
    except ImportError:
        return error_response(
            code="shap_not_available",
            message="La biblioteca SHAP no está instalada en el servidor.",
            model_id=manifest.model_id,
        )
    except Exception as exc:
        logger.exception(f"Error al calcular SHAP para {target}/{algorithm}: {exc}")
        return error_response(
            code="explain_error",
            message=f"No fue posible calcular la explicabilidad: {exc}",
            model_id=manifest.model_id,
        )

    response = ExplainResponse(
        contributions=[
            ShapContributionSchema(
                feature=c.feature,
                value=c.value,
                shap_value=c.shap_value,
            )
            for c in contributions
        ],
        top_n=top_n,
        algorithm=manifest.algorithm,
        model_id=manifest.model_id,
    )
    return success_response(response.model_dump(mode="json"), model_id=manifest.model_id)
