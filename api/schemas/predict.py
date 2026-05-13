from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from api.domain.manifest import ModelManifest


class PredictResponse(BaseModel):
    predicted_class: int
    probability: float
    threshold: float
    risk_level: str
    calibrated: bool
    prevalence_train: float | None
    warnings: list[str]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n: int


def _python_type_for(dtype: str) -> type:
    """Mapea dtypes pandas/numpy a tipos Python aceptables vía JSON."""
    d = dtype.lower()
    if d.startswith(("int", "uint")):
        return int
    if d.startswith("bool"):
        return bool
    if d.startswith("float"):
        return float
    return float


def build_predict_request_validator(manifest: ModelManifest) -> type[BaseModel]:
    feature_fields: dict[str, tuple[type, Any]] = {}
    for name in manifest.feature_names:
        py_type = _python_type_for(manifest.feature_dtypes.get(name, "float64"))
        feature_fields[name] = (py_type | None, Field(default=None))

    FeaturesModel = create_model(
        f"FeaturesFor_{manifest.algorithm}_{manifest.target_version}",
        __config__=ConfigDict(extra="forbid"),
        **feature_fields,
    )

    PredictRequestModel = create_model(
        f"PredictRequest_{manifest.algorithm}_{manifest.target_version}",
        __config__=ConfigDict(extra="forbid"),
        features=(FeaturesModel, ...),
    )
    return PredictRequestModel


def build_batch_predict_request_validator(manifest: ModelManifest) -> type[BaseModel]:
    """Versión batch: lista de FeaturesModel."""
    feature_fields: dict[str, tuple[type, Any]] = {}
    for name in manifest.feature_names:
        py_type = _python_type_for(manifest.feature_dtypes.get(name, "float64"))
        feature_fields[name] = (py_type | None, Field(default=None))

    FeaturesModel = create_model(
        f"FeaturesFor_{manifest.algorithm}_{manifest.target_version}_Batch",
        __config__=ConfigDict(extra="forbid"),
        **feature_fields,
    )

    BatchPredictRequestModel = create_model(
        f"BatchPredictRequest_{manifest.algorithm}_{manifest.target_version}",
        __config__=ConfigDict(extra="forbid"),
        items=(list[FeaturesModel], ...),
    )
    return BatchPredictRequestModel


__all__ = [
    "PredictResponse",
    "BatchPredictResponse",
    "build_predict_request_validator",
    "build_batch_predict_request_validator",
    "ValidationError",
]
