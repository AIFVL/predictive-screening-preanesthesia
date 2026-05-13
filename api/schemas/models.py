from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TargetInfo(BaseModel):
    slug: str
    display_name: str
    description: str
    recommended: bool
    n_models: int


class ModelSummary(BaseModel):
    target: str
    target_display_name: str
    algorithm: str
    model_id: str
    calibrated: bool
    calibration_method: str | None
    threshold: float
    performance: dict[str, float | None]
    warnings: list[str]
    recommended: bool


class FeatureSpec(BaseModel):
    name: str
    dtype: str
    required: bool
    median: float | None
    description: str | None = None


class ModelSchema(BaseModel):
    model_id: str
    target: str
    algorithm: str
    features: list[FeatureSpec]
    threshold: float
    threshold_metric: str
    prevalence: dict[str, Any]
    calibrated: bool
    calibration_method: str | None
    imputation: dict[str, Any]
    warnings: list[str]


class ModelDetail(BaseModel):
    model_id: str
    target: str
    target_display_name: str
    algorithm: str
    calibrated: bool
    calibration: dict[str, Any]
    threshold: float
    threshold_metric: str
    prevalence: dict[str, Any]
    performance: dict[str, float | None]
    warnings: list[str]
    schema_version: str
    created_at: str | None
    n_features: int
