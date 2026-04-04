# src/models/trainer.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.registry import build_model
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.utils.io import write_json, write_joblib, read_joblib
from src.utils.logger import get_logger

logger = get_logger("models.trainer")


def train_model(
    model_cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path | str,
    model_name: str,
    random_state: int = 42,
    val_size: float = 0.2,
    threshold_metric: str = "f2",
    optimize_for: str | None = None,
    recall_min: float = 0.80,
) -> dict:
    """
    Entrena un modelo, encuentra threshold óptimo, serializa joblib y escribe métricas JSON.

    Usa un split interno train/val para encontrar el threshold.
    Entrena el modelo final en X completo.

    Retorna artifact dict con: model_path, metrics_path, threshold, metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y))
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = build_model(model_cfg, random_state=random_state)

    logger.info(f"[{model_name}] Entrenando en {len(X_train):,} filas...")
    model.fit(X_train.apply(pd.to_numeric, errors="coerce").fillna(-1), y_train)

    X_val_num = X_val.apply(pd.to_numeric, errors="coerce").fillna(-1)
    y_val_proba = model.predict_proba(X_val_num)[:, 1]

    threshold, _, _, _ = find_optimal_threshold(
        y_val, y_val_proba,
        metric=threshold_metric,
        optimize_for=optimize_for,
        recall_min=recall_min,
    )

    y_val_pred = (y_val_proba >= threshold).astype(int)
    metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba, threshold)

    # Re-entrenar en datos completos
    model_final = build_model(model_cfg, random_state=random_state)
    model_final.fit(X.apply(pd.to_numeric, errors="coerce").fillna(-1), y)

    model_path = out_dir / f"{model_name}_model.joblib"
    metrics_path = out_dir / f"{model_name}_metrics.json"

    write_joblib(model_final, model_path)
    write_json({**metrics, "threshold": threshold, "model_name": model_name}, metrics_path)

    logger.info(
        f"[{model_name}] threshold={threshold:.3f} | "
        f"Recall={metrics['Recall']:.3f} | F2={metrics['F2']:.3f} | ROC_AUC={metrics['ROC_AUC']:.3f}"
    )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "threshold": threshold,
        "metrics": metrics,
    }


def load_model(model_path: Path | str):
    """Carga un modelo serializado desde joblib."""
    return read_joblib(Path(model_path))
