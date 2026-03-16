"""
Feature Processor
=================
Converts a raw patient dict into a single-row DataFrame ready for model
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENCODING_FIX_MAP: dict[str, str] = {
    "PrÃ³tesis Dental_movil": "Prótesis Dental_movil",
    "PrÃ³tesis Dental_no": "Prótesis Dental_no",
    "AlÃ©rgeno_med_opioides": "Alérgeno_med_opioides",
    "AlÃ©rgeno_suplementos": "Alérgeno_suplementos",
    "TensiÃ³n Arterial SistÃ³lica (mm/Hg)": "Tensión Arterial Sistólica (mm/Hg)",
    "TensiÃ³n Arterial DiastÃ³lica (mm/Hg)": "Tensión Arterial Diastólica (mm/Hg)",
    "TensiÃ³n Arterial Media (mm/Hg)": "Tensión Arterial Media (mm/Hg)",
    "Prote\xc3\xadna C reactiva (mg/l)": "Proteína C reactiva (mg/l)",
}

_REVERSE_ENCODING_FIX_MAP: dict[str, str] = {v: k for k, v in ENCODING_FIX_MAP.items()}


def _fix_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Apply ENCODING_FIX_MAP to incoming patient data keys."""
    return {ENCODING_FIX_MAP.get(k, k): v for k, v in data.items()}


def process_patient(
    patient_data: dict[str, Any],
    artifact: dict,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Transform a raw patient dict into a model-ready DataFrame.

    Parameters
    ----------
    patient_data : dict
        Raw feature key-value pairs from the API request.
    artifact : dict
        Loaded model artifact (see model_registry.py for expected keys).

    Returns
    -------
    X : pd.DataFrame
        Single-row DataFrame aligned to artifact["feature_columns"].
    missing_features : list[str]
        Features that were absent in the request and were imputed.
    warnings : list[str]
        Non-fatal notices (e.g., unknown features ignored).
    """
    warnings: list[str] = []
    missing_features: list[str] = []

    data = _fix_keys(patient_data)

    feature_columns: list[str] = artifact["feature_columns"]
    training_medians: dict[str, float] = artifact.get("training_medians", {})
    strategies: dict[str, list[str]] = artifact.get(
        "imputation_strategies", {"fill_zero": [], "fill_median": []}
    )
    fill_zero_set = set(strategies.get("fill_zero", []))
    fill_median_set = set(strategies.get("fill_median", []))

    known = set(feature_columns)
    unknown = set(data.keys()) - known - {"Edad"}
    if unknown:
        warnings.append(
            f"Ignoring {len(unknown)} unknown feature(s): {sorted(unknown)[:5]}"
            + (" …" if len(unknown) > 5 else "")
        )

    row: dict[str, float] = {}
    for col in feature_columns:
        if col in data and data[col] is not None:
            try:
                row[col] = float(data[col])
            except (TypeError, ValueError):
                warnings.append(
                    f"Feature '{col}' has non-numeric value '{data[col]}'; imputing."
                )
                row[col] = _impute_single(col, fill_zero_set, fill_median_set, training_medians)
                missing_features.append(col)
        else:
            row[col] = _impute_single(col, fill_zero_set, fill_median_set, training_medians)
            missing_features.append(col)

    X = pd.DataFrame([row], columns=feature_columns)

    preprocessor = artifact.get("preprocessor")
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X)
            if isinstance(X_transformed, np.ndarray):
                X = pd.DataFrame(X_transformed, columns=feature_columns)
            else:
                X = X_transformed
        except Exception as exc:
            logger.warning("Preprocessor transform failed: %s. Using raw features.", exc)
            warnings.append(f"Preprocessor skipped ({exc}); raw features used.")

    return X, missing_features, warnings


def process_batch(
    patients: list[dict[str, Any]],
    artifact: dict,
) -> tuple[pd.DataFrame, list[list[str]], list[list[str]]]:
    """
    Process a list of patient dicts into a multi-row DataFrame.

    Returns
    -------
    X : pd.DataFrame
        Shape (n_patients, n_features).
    missing_per_patient : list[list[str]]
    warnings_per_patient : list[list[str]]
    """
    rows, missing_per_patient, warnings_per_patient = [], [], []
    feature_columns: list[str] = artifact["feature_columns"]
    training_medians = artifact.get("training_medians", {})
    strategies = artifact.get("imputation_strategies", {"fill_zero": [], "fill_median": []})
    fill_zero_set = set(strategies.get("fill_zero", []))
    fill_median_set = set(strategies.get("fill_median", []))

    for patient_data in patients:
        data = _fix_keys(patient_data)
        warnings: list[str] = []
        missing: list[str] = []
        row: dict[str, float] = {}

        for col in feature_columns:
            if col in data and data[col] is not None:
                try:
                    row[col] = float(data[col])
                except (TypeError, ValueError):
                    row[col] = _impute_single(col, fill_zero_set, fill_median_set, training_medians)
                    missing.append(col)
                    warnings.append(f"Feature '{col}' non-numeric; imputed.")
            else:
                row[col] = _impute_single(col, fill_zero_set, fill_median_set, training_medians)
                missing.append(col)

        rows.append(row)
        missing_per_patient.append(missing)
        warnings_per_patient.append(warnings)

    X = pd.DataFrame(rows, columns=feature_columns)

    preprocessor = artifact.get("preprocessor")
    if preprocessor is not None:
        try:
            X_transformed = preprocessor.transform(X)
            if isinstance(X_transformed, np.ndarray):
                X = pd.DataFrame(X_transformed, columns=feature_columns)
            else:
                X = X_transformed
        except Exception as exc:
            logger.warning("Batch preprocessor failed: %s. Using raw features.", exc)
            for w in warnings_per_patient:
                w.append(f"Preprocessor skipped ({exc}).")

    return X, missing_per_patient, warnings_per_patient


def _impute_single(
    col: str,
    fill_zero_set: set[str],
    fill_median_set: set[str],
    training_medians: dict[str, float],
) -> float:
    """Return imputed value for a single missing feature."""
    if col in fill_zero_set:
        return 0.0
    if col in fill_median_set:
        return float(training_medians.get(col, 0.0))
    return 0.0


def check_age_filter(patient_data: dict[str, Any], min_age: int) -> str | None:
    """Return a warning message if the patient's age violates the filter."""
    data = _fix_keys(patient_data)
    age = data.get("Edad") or patient_data.get("Edad") or patient_data.get("edad")
    if age is not None:
        try:
            if float(age) < min_age:
                return (
                    f"Patient age ({age}) is below the model's minimum age ({min_age}). "
                    "This model was trained on adult patients only; results may be unreliable."
                )
        except (TypeError, ValueError):
            pass
    return None
