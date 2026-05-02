# src/models/trainer.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.models.manifest import build_manifest, manifest_path_for, write_manifest
from src.models.registry import fit_with_calibration
from src.utils.io import read_joblib, write_joblib, write_json
from src.utils.logger import get_logger

logger = get_logger("models.trainer")


def _to_numeric_filled(df: pd.DataFrame) -> pd.DataFrame:
    """Imputación canónica del proyecto: coerción a numérico + fillna(-1)."""
    return df.apply(pd.to_numeric, errors="coerce").fillna(-1)


def train_model(
    model_cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path | str,
    model_name: str,
    random_state: int = 42,
    val_size: float = 0.2,
    threshold_metric: str = "f2",
    optimize_for: str | None = "recall_constraint",
    recall_min: float = 0.85,
) -> dict:
    """
    Entrena un modelo intentando calibrarlo, encuentra threshold óptimo (post-calibración),
    serializa el modelo en joblib y escribe métricas + manifest JSON.

    Pipeline:
      1. Split interno train/val sobre X (estratificado).
      2. Ajusta el modelo en train_inner con `fit_with_calibration` (intenta isotonic/sigmoid
         según algoritmo; si todo falla, queda sin calibrar y emite warning).
      3. Encuentra threshold óptimo sobre val_inner usando las probabilidades ya calibradas.
      4. Re-entrena el modelo final en X completo (también calibrado).
      5. Persiste: `<model_name>_model.joblib`, `<model_name>_metrics.json`, `<model_name>_manifest.json`.

    Retorna artifact dict con: model_path, metrics_path, manifest_path, threshold, metrics,
    y `calibration` con el estado de calibración del modelo persistido.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y))
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    X_train_num = _to_numeric_filled(X_train)
    X_val_num = _to_numeric_filled(X_val)

    logger.info(f"[{model_name}] Entrenando (con calibración) en {len(X_train):,} filas...")
    model_inner, calib_inner = fit_with_calibration(
        model_cfg, X_train_num, y_train, random_state=random_state
    )

    y_val_proba = model_inner.predict_proba(X_val_num)[:, 1]

    threshold, _, _, _ = find_optimal_threshold(
        y_val, y_val_proba,
        metric=threshold_metric,
        optimize_for=optimize_for,
        recall_min=recall_min,
    )

    y_val_pred = (y_val_proba >= threshold).astype(int)
    metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba, threshold)

    logger.info(f"[{model_name}] Re-entrenando en X completo ({len(X):,} filas) para persistencia...")
    X_num = _to_numeric_filled(X)
    model_final, calib_final = fit_with_calibration(
        model_cfg, X_num, y, random_state=random_state
    )

    model_path = out_dir / f"{model_name}_model.joblib"
    metrics_path = out_dir / f"{model_name}_metrics.json"
    manifest_path = manifest_path_for(out_dir, model_name)

    write_joblib(model_final, model_path)
    write_json(
        {**metrics, "threshold": threshold, "model_name": model_name},
        metrics_path,
    )

    target_version = out_dir.name
    extra_warnings: list[str] = []
    if calib_inner.get("method") and calib_final.get("method") and calib_inner["method"] != calib_final["method"]:
        extra_warnings.append("calibration_method_changed_between_inner_and_final_fit")

    manifest = build_manifest(
        target_version=target_version,
        algorithm=model_name,
        model_filename=model_path.name,
        metrics_filename=metrics_path.name,
        X_train=X,
        y_train=y,
        threshold=threshold,
        threshold_metric=threshold_metric,
        calibration_info=calib_final,
        metrics=metrics,
        extra_warnings=extra_warnings,
    )
    write_manifest(manifest, manifest_path)

    logger.info(
        f"[{model_name}] threshold={threshold:.3f} | "
        f"calibrated={calib_final['calibrated']} ({calib_final.get('method')}) | "
        f"Recall={metrics['Recall']:.3f} | F2={metrics['F2']:.3f} | ROC_AUC={metrics['ROC_AUC']:.3f}"
    )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "threshold": threshold,
        "metrics": metrics,
        "calibration": calib_final,
    }


def load_model(model_path: Path | str):
    """Carga un modelo serializado desde joblib."""
    return read_joblib(Path(model_path))
