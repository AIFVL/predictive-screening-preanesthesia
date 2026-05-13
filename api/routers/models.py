from fastapi import APIRouter, HTTPException, Request

from api.domain.manifest import ModelManifest
from api.domain.registry import ModelRegistry
from api.schemas.common import success_response
from api.schemas.models import FeatureSpec, ModelDetail, ModelSchema

router = APIRouter(prefix="/models", tags=["models"])


def _require_manifest(registry: ModelRegistry, target: str, algorithm: str) -> ModelManifest:
    manifest = registry.get_manifest(target, algorithm)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado: target='{target}', algorithm='{algorithm}'",
        )
    return manifest


@router.get("")
async def list_models(request: Request) -> dict:
    """Lista plana de todos los modelos servidos."""
    registry: ModelRegistry = request.app.state.registry
    return success_response(registry.list_models())


@router.get("/{target}/{algorithm}")
async def get_model(target: str, algorithm: str, request: Request) -> dict:
    """Metadata completa de un modelo."""
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)
    alias = registry.alias(target)
    detail = ModelDetail(
        model_id=manifest.model_id,
        target=target,
        target_display_name=alias.display_name if alias else target,
        algorithm=manifest.algorithm,
        calibrated=manifest.calibrated,
        calibration=manifest.calibration,
        threshold=manifest.threshold,
        threshold_metric=manifest.threshold_metric,
        prevalence=manifest.prevalence,
        performance=manifest.performance,
        warnings=manifest.warnings,
        schema_version=manifest.schema_version,
        created_at=manifest.created_at,
        n_features=len(manifest.feature_names),
    )
    return success_response(detail.model_dump(mode="json"), model_id=manifest.model_id)


@router.get("/{target}/{algorithm}/schema")
async def get_model_schema(target: str, algorithm: str, request: Request) -> dict:
    """
    Schema accionable para el frontend: lista de features con dtype y mediana
    (sirve como default para inputs numéricos).
    """
    registry: ModelRegistry = request.app.state.registry
    manifest = _require_manifest(registry, target, algorithm)

    features = [
        FeatureSpec(
            name=name,
            dtype=manifest.feature_dtypes.get(name, "float64"),
            required=False,  # todas opcionales — el preprocessor imputa
            median=manifest.feature_medians.get(name),
        )
        for name in manifest.feature_names
    ]
    schema = ModelSchema(
        model_id=manifest.model_id,
        target=target,
        algorithm=manifest.algorithm,
        features=features,
        threshold=manifest.threshold,
        threshold_metric=manifest.threshold_metric,
        prevalence=manifest.prevalence,
        calibrated=manifest.calibrated,
        calibration_method=manifest.calibration.get("method"),
        imputation=manifest.imputation,
        warnings=manifest.warnings,
    )
    return success_response(schema.model_dump(mode="json"), model_id=manifest.model_id)
