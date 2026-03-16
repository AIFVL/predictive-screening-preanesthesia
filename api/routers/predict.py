"""
Prediction endpoints
====================
GET  /predict/schema/{model_id}  - required feature schema for a model
POST /predict                    - single patient prediction
POST /predict/batch              - batch prediction (up to max_batch_size)
"""

import logging

from fastapi import APIRouter, HTTPException

from api.core.config import settings
from api.schemas.requests import BatchPredictRequest, PredictRequest
from api.schemas.responses import (
    BatchPredictResponse,
    FeatureSchema,
    ModelSchemaResponse,
    PredictResponse,
)
from api.services.model_registry import registry
from api.services.predictor import predict_batch, predict_single

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.get(
    "/schema/{model_id}",
    response_model=ModelSchemaResponse,
    summary="Get required features for a model",
)
def get_schema(model_id: str):
    """
    Returns the ordered list of features expected by the model, along with
    each feature's imputation strategy and training-set median.

    Use this to build a valid prediction request body.
    """
    try:
        artifact = registry.get_artifact(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    feature_columns: list[str] = artifact["feature_columns"]
    training_medians: dict = artifact.get("training_medians", {})
    strategies: dict = artifact.get(
        "imputation_strategies", {"fill_zero": [], "fill_median": []}
    )
    fill_zero_set = set(strategies.get("fill_zero", []))
    fill_median_set = set(strategies.get("fill_median", []))

    features = [
        FeatureSchema(
            name=col,
            imputation_strategy=(
                "fill_zero"
                if col in fill_zero_set
                else "fill_median"
                if col in fill_median_set
                else "fill_zero"
            ),
            training_median=training_medians.get(col),
        )
        for col in feature_columns
    ]

    return ModelSchemaResponse(
        model_id=model_id,
        version=artifact.get("version", "unknown"),
        n_features=len(feature_columns),
        features=features,
        min_age=artifact.get("min_age", 18),
    )


@router.post(
    "",
    response_model=PredictResponse,
    summary="Single patient prediction",
)
def predict(body: PredictRequest):
    """
    Predict whether a single patient requires preanesthesia evaluation.

    - **model_id**: Use `GET /models` to see available model IDs.
    - **patient_data**: Key-value pairs. Use `GET /predict/schema/{model_id}` for
      the full feature list. Missing features are imputed automatically.
    - **return_explanations**: Set to `true` to receive SHAP feature attributions.

    Returns the predicted class (0/1), probability, risk level, and recommendation.
    """
    if body.model_id not in registry.list_ids():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model_id}' not found. "
            f"Available: {registry.list_ids()}",
        )

    try:
        return predict_single(
            model_id=body.model_id,
            patient_data=body.patient_data,
            return_explanations=body.return_explanations,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch patient prediction",
)
def predict_batch_endpoint(body: BatchPredictRequest):
    """
    Predict for multiple patients in a single request.

    - Maximum batch size is controlled by the `API_MAX_BATCH_SIZE` env variable
      (default: 100).
    - SHAP explanations in batch mode add significant latency; use with care.

    Returns a result per patient with index, prediction, warnings, and any
    imputed features.
    """
    if len(body.patients) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(body.patients)} exceeds maximum {settings.max_batch_size}.",
        )

    if body.model_id not in registry.list_ids():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model_id}' not found. "
            f"Available: {registry.list_ids()}",
        )

    try:
        return predict_batch(
            model_id=body.model_id,
            patients=body.patients,
            return_explanations=body.return_explanations,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
