from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger("models.manifest")

MANIFEST_SCHEMA_VERSION = "1.0"


def _to_jsonable(value: Any) -> Any:
    """Convierte tipos numpy/pandas a equivalentes JSON-serializables."""
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        if isinstance(value, np.floating) and (np.isnan(value) or np.isinf(value)):
            return None
        return value.item()
    if isinstance(value, (np.ndarray, pd.Series)):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _compute_feature_metadata(X: pd.DataFrame) -> dict[str, Any]:
    """Extrae feature_names, dtypes y medianas (post-coerción a numérico)."""
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    medians = X_numeric.median(numeric_only=True)
    return {
        "feature_names": list(X.columns),
        "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        "feature_medians": {
            col: _to_jsonable(medians[col]) if col in medians.index else None
            for col in X.columns
        },
    }


def build_manifest(
    *,
    target_version: str,
    algorithm: str,
    model_filename: str,
    metrics_filename: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    threshold_metric: str,
    calibration_info: dict,
    metrics: dict,
    extra_warnings: list[str] | None = None,
) -> dict:
    """
    Construye el manifest a partir de los artefactos de entrenamiento.

    Parámetros:
        target_version: nombre interno del target (ej. 'target_f_predictibilidad_maxima').
        algorithm: nombre del algoritmo (ej. 'xgboost').
        model_filename: nombre relativo del .joblib (ej. 'xgboost_model.joblib').
        metrics_filename: nombre relativo del *_metrics.json.
        X_train: features usadas para entrenar el modelo final.
        y_train: target binario.
        threshold: threshold óptimo (post-calibración).
        threshold_metric: métrica que se optimizó para encontrar el threshold (ej. 'f2').
        calibration_info: dict devuelto por `fit_with_calibration`.
        metrics: dict de métricas test/val (output de compute_classification_metrics).
        extra_warnings: warnings adicionales a incluir (además de los de calibración).
    """
    feature_meta = _compute_feature_metadata(X_train)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    n_total = n_pos + n_neg
    prevalence_train = (n_pos / n_total) if n_total > 0 else None

    warnings_list = list(calibration_info.get("warnings", []))
    if extra_warnings:
        warnings_list.extend(extra_warnings)

    performance = {
        "roc_auc": _to_jsonable(metrics.get("ROC_AUC")),
        "pr_auc": _to_jsonable(metrics.get("PR_AUC")),
        "f2": _to_jsonable(metrics.get("F2")),
        "f1": _to_jsonable(metrics.get("F1")),
        "recall": _to_jsonable(metrics.get("Recall")),
        "precision": _to_jsonable(metrics.get("Precision")),
        "specificity": _to_jsonable(metrics.get("Specificity")),
        "balanced_accuracy": _to_jsonable(metrics.get("Balanced_Accuracy")),
        "brier": _to_jsonable(metrics.get("Brier")),
        "accuracy": _to_jsonable(metrics.get("Accuracy")),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_id": f"{target_version}__{algorithm}",
        "target_version": target_version,
        "algorithm": algorithm,
        "model_filename": model_filename,
        "metrics_filename": metrics_filename,
        "feature_names": feature_meta["feature_names"],
        "feature_dtypes": feature_meta["feature_dtypes"],
        "feature_medians": feature_meta["feature_medians"],
        "imputation": {"strategy": "fill_constant", "value": -1},
        "threshold": _to_jsonable(threshold),
        "threshold_metric": threshold_metric,
        "calibrated": bool(calibration_info.get("calibrated", False)),
        "calibration": {
            "method": calibration_info.get("method"),
            "cv": calibration_info.get("cv"),
            "fallback_reason": calibration_info.get("fallback_reason"),
        },
        "prevalence": {
            "train": _to_jsonable(prevalence_train),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": n_total,
        },
        "performance": performance,
        "warnings": warnings_list,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return manifest


def write_manifest(manifest: dict, out_path: Path | str) -> Path:
    """Escribe el manifest a disco como JSON pretty-printed."""
    out_path = Path(out_path)
    write_json(_to_jsonable(manifest), out_path)
    logger.info(
        f"Manifest escrito en {out_path} "
        f"(calibrated={manifest['calibrated']}, "
        f"method={manifest['calibration']['method']})"
    )
    return out_path


def manifest_path_for(model_dir: Path | str, model_name: str) -> Path:
    """Convención: <model_dir>/<model_name>_manifest.json"""
    return Path(model_dir) / f"{model_name}_manifest.json"
