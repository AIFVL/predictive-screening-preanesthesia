"""
Predictor
=========
Wraps the feature processor + model inference + optional SHAP explanations
into a single, reusable prediction call.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from api.core.config import settings
from api.schemas.responses import (
    BatchPredictItem,
    BatchPredictResponse,
    ExplanationItem,
    Prediction,
    PredictResponse,
)
from api.services.feature_processor import check_age_filter, process_batch, process_patient
from api.services.model_registry import registry

logger = logging.getLogger(__name__)

# ── SHAP explainer cache (per model_id) ───────────────────────────────────────
_shap_explainer_cache: dict[str, Any] = {}


def _get_shap_explainer(model_id: str, artifact: dict, X_background: pd.DataFrame | None = None):
    """Return a cached SHAP explainer, creating one if needed."""
    if model_id in _shap_explainer_cache:
        return _shap_explainer_cache[model_id]

    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; explanations disabled.")
        return None

    model = artifact["model"]
    model_type = artifact.get("model_type", "tree")

    try:
        if model_type in {"tree", "ensemble"}:
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            bg = X_background if X_background is not None else np.zeros((1, len(artifact["feature_columns"])))
            explainer = shap.LinearExplainer(model, bg)
        else:
            logger.warning(
                "Model type '%s' uses KernelExplainer; SHAP may be slow.", model_type
            )
            bg = X_background if X_background is not None else np.zeros((1, len(artifact["feature_columns"])))
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        _shap_explainer_cache[model_id] = explainer
        return explainer

    except Exception as exc:
        logger.warning("Could not build SHAP explainer for '%s': %s", model_id, exc)
        return None


def _compute_shap(
    model_id: str,
    artifact: dict,
    X: pd.DataFrame,
    top_n: int = 10,
) -> list[ExplanationItem] | None:
    """Compute SHAP values for X and return top-N features by |shap_value|."""
    explainer = _get_shap_explainer(model_id, artifact)
    if explainer is None:
        return None

    try:
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        feature_columns = artifact["feature_columns"]
        explanations: list[ExplanationItem] = []

        for i, col in enumerate(feature_columns):
            sv = float(shap_values[0, i]) if X.shape[0] == 1 else float(shap_values[:, i].mean())
            explanations.append(
                ExplanationItem(
                    feature=col,
                    patient_value=float(X.iloc[0][col]) if X.shape[0] == 1 else None,
                    shap_value=sv,
                    direction="increases_risk" if sv > 0 else "decreases_risk",
                )
            )

        explanations.sort(key=lambda e: abs(e.shap_value), reverse=True)
        return explanations[:top_n]

    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


# ── Risk level helper ─────────────────────────────────────────────────────────

def _risk_level(probability: float, threshold: float) -> str:
    """Map probability to low / medium / high relative to the model threshold."""
    if probability < threshold * 0.5:
        return "low"
    if probability < threshold:
        return "medium"
    if probability < threshold + (1.0 - threshold) * 0.5:
        return "medium"
    return "high"


def _recommendation(predicted_class: int, risk_level: str) -> str:
    if predicted_class == 1:
        return "SCHEDULE PREANESTHESIA EVALUATION"
    if risk_level == "medium":
        return "BORDERLINE – consider clinical review"
    return "PREANESTHESIA EVALUATION NOT INDICATED"


# ── Public API ────────────────────────────────────────────────────────────────

def predict_single(
    model_id: str,
    patient_data: dict[str, Any],
    return_explanations: bool = False,
) -> PredictResponse:
    artifact = registry.get_artifact(model_id)
    model = artifact["model"]
    threshold: float = artifact["optimal_threshold"]
    min_age: int = artifact.get("min_age", 18)

    warnings: list[str] = []

    age_warn = check_age_filter(patient_data, min_age)
    if age_warn:
        warnings.append(age_warn)
    X, missing_features, proc_warnings = process_patient(patient_data, artifact)
    warnings.extend(proc_warnings)

    try:
        proba: float = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        raise RuntimeError(f"Model inference failed for '{model_id}': {exc}") from exc

    predicted_class = int(proba >= threshold)
    risk = _risk_level(proba, threshold)

    explanations = None
    if return_explanations and settings.enable_shap:
        explanations = _compute_shap(model_id, artifact, X)

    prediction = Prediction(
        predicted_class=predicted_class,
        probability=round(proba, 6),
        risk_level=risk,
        threshold_used=threshold,
        recommendation=_recommendation(predicted_class, risk),
        model_id=model_id,
        version=artifact.get("version", "unknown"),
        explanations=explanations,
    )

    return PredictResponse(
        prediction=prediction,
        missing_features=missing_features,
        warnings=warnings,
    )


def predict_batch(
    model_id: str,
    patients: list[dict[str, Any]],
    return_explanations: bool = False,
) -> BatchPredictResponse:
    if len(patients) > settings.max_batch_size:
        raise ValueError(
            f"Batch size {len(patients)} exceeds the maximum of {settings.max_batch_size}."
        )

    artifact = registry.get_artifact(model_id)
    model = artifact["model"]
    threshold: float = artifact["optimal_threshold"]
    min_age: int = artifact.get("min_age", 18)

    X_all, missing_per, warnings_per = process_batch(patients, artifact)

    try:
        probas: np.ndarray = model.predict_proba(X_all)[:, 1]
    except Exception as exc:
        raise RuntimeError(f"Batch inference failed for '{model_id}': {exc}") from exc

    results: list[BatchPredictItem] = []

    for idx, (proba, missing, warns, patient_data) in enumerate(
        zip(probas, missing_per, warnings_per, patients)
    ):
        proba = float(proba)
        age_warn = check_age_filter(patient_data, min_age)
        if age_warn:
            warns.append(age_warn)

        predicted_class = int(proba >= threshold)
        risk = _risk_level(proba, threshold)

        explanations = None
        if return_explanations and settings.enable_shap:
            X_row = X_all.iloc[[idx]]
            explanations = _compute_shap(model_id, artifact, X_row)

        prediction = Prediction(
            predicted_class=predicted_class,
            probability=round(proba, 6),
            risk_level=risk,
            threshold_used=threshold,
            recommendation=_recommendation(predicted_class, risk),
            model_id=model_id,
            version=artifact.get("version", "unknown"),
            explanations=explanations,
        )

        results.append(
            BatchPredictItem(
                patient_index=idx,
                prediction=prediction,
                missing_features=missing,
                warnings=warns,
            )
        )

    successful = sum(1 for r in results if r.prediction is not None)
    return BatchPredictResponse(
        model_id=model_id,
        total=len(patients),
        successful=successful,
        failed=len(patients) - successful,
        results=results,
    )
