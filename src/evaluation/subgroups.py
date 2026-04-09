# src/evaluation/subgroups.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold
from src.utils.logger import get_logger

logger = get_logger("evaluation.subgroups")


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    optimize_for: str = "recall_constraint",
    recall_min: float = 0.85,
) -> dict:
    """
    Ejecuta cross-validation estratificada con threshold óptimo por fold.

    Cada fold busca su propio threshold usando find_optimal_threshold con los
    mismos parámetros que el entrenamiento principal, garantizando que las
    métricas CV sean comparables con las métricas del test set.

    Retorna dict con mean y std de: ROC_AUC, F2, Recall, Precision, Specificity, Balanced_Accuracy.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(-1)

    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_num, y)):
        X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]

        threshold, _, _, _ = find_optimal_threshold(
            y_val, y_proba,
            optimize_for=optimize_for,
            recall_min=recall_min,
        )
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_classification_metrics(y_val, y_pred, y_proba, threshold)
        fold_metrics.append(metrics)

    result = {}
    for metric in ["ROC_AUC", "F2", "Recall", "Precision", "Specificity", "Balanced_Accuracy"]:
        vals = [f[metric] for f in fold_metrics]
        result[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        result[f"{metric}_std"] = round(float(np.std(vals)), 4)

    logger.info(
        f"CV {n_folds}-fold: ROC_AUC={result['ROC_AUC_mean']:.3f}±{result['ROC_AUC_std']:.3f} | "
        f"Recall={result['Recall_mean']:.3f}±{result['Recall_std']:.3f} | "
        f"Specificity={result['Specificity_mean']:.3f}±{result['Specificity_std']:.3f}"
    )
    return result
