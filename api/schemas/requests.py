from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Single-patient prediction request."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(
        ...,
        description="Model ID (artifact filename without .joblib). "
        "Use GET /models to list available IDs.",
        examples=["xgboost_target_b"],
    )
    patient_data: dict[str, Any] = Field(
        ...,
        description="Patient features as key-value pairs. "
        "Use GET /predict/schema/{model_id} to see required fields.",
        examples=[{"Edad": 45, "Tensión Arterial Sistólica (mm/Hg)": 120}],
    )
    return_explanations: bool = Field(
        False,
        description="Include SHAP feature-level explanations in the response. "
        "Adds latency for non-tree models.",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request (up to max_batch_size patients)."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="Model ID to use for all patients.")
    patients: list[dict[str, Any]] = Field(
        ...,
        description="List of patient feature dicts.",
        min_length=1,
    )
    return_explanations: bool = Field(
        False,
        description="Include SHAP explanations per patient. "
        "Not recommended for large batches.",
    )
