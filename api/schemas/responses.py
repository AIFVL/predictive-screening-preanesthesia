from typing import Any
from pydantic import BaseModel, ConfigDict, Field

_no_ns = ConfigDict(protected_namespaces=())


# ── Model registry responses ──────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    """Validation metrics stored inside the artifact."""

    roc_auc: float | None = None
    pr_auc: float | None = None
    recall: float | None = None
    precision: float | None = None
    f1: float | None = None
    f2: float | None = None
    accuracy: float | None = None
    specificity: float | None = None
    brier_score: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelInfo(BaseModel):
    """Summary of a loaded model artifact."""

    model_config = _no_ns
    model_id: str
    model_name: str
    model_type: str
    version: str
    description: str
    n_features: int
    optimal_threshold: float
    min_age: int
    created_at: str
    metrics: ModelMetrics
    top_features: list[FeatureImportanceItem] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    total: int


class ModelUploadResponse(BaseModel):
    model_config = _no_ns
    model_id: str
    message: str
    model_info: ModelInfo


class ModelDeleteResponse(BaseModel):
    model_config = _no_ns
    model_id: str
    message: str


# ── Prediction responses ──────────────────────────────────────────────────────

class ExplanationItem(BaseModel):
    feature: str
    patient_value: Any
    shap_value: float
    direction: str


class Prediction(BaseModel):
    """Core prediction result for a single patient."""

    model_config = _no_ns
    predicted_class: int = Field(..., description="0 = no evaluation needed, 1 = evaluation needed")
    probability: float = Field(..., description="Model's estimated probability [0, 1]")
    risk_level: str = Field(..., description="low | medium | high")
    threshold_used: float
    recommendation: str

    model_id: str
    version: str
    explanations: list[ExplanationItem] | None = None


class PredictResponse(BaseModel):
    prediction: Prediction
    missing_features: list[str] = Field(
        default_factory=list,
        description="Features not supplied by caller; imputed automatically.",
    )
    warnings: list[str] = Field(default_factory=list)


class BatchPredictItem(BaseModel):
    patient_index: int
    prediction: Prediction | None = None
    error: str | None = None
    missing_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BatchPredictResponse(BaseModel):
    model_config = _no_ns
    model_id: str
    total: int
    successful: int
    failed: int
    results: list[BatchPredictItem]


# ── Schema endpoint response ──────────────────────────────────────────────────

class FeatureSchema(BaseModel):
    name: str
    imputation_strategy: str
    training_median: float | None = None


class ModelSchemaResponse(BaseModel):
    model_config = _no_ns
    model_id: str
    version: str
    n_features: int
    features: list[FeatureSchema]
    min_age: int
    note: str = (
        "Missing numeric features are imputed automatically. "
        "Apply ENCODING_FIX_MAP for Spanish column names with encoding issues."
    )


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    version: str
    shap_enabled: bool
