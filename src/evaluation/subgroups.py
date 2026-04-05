# src/evaluation/subgroups.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import find_optimal_threshold
from src.utils.logger import get_logger

logger = get_logger("evaluation.subgroups")


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    optimize_for: str | None = None,
    recall_min: float = 0.85,
) -> dict:
    """
    Ejecuta cross-validation estratificada y retorna estadísticas de métricas.

    Retorna dict con mean y std de: ROC_AUC, F2, Recall, Precision, Specificity.
    Cuando optimize_for='recall_constraint', usa find_optimal_threshold para
    determinar el umbral de clasificación en cada fold.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(-1)

    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_num, y)):
        X_tr, X_val = X_num.iloc[train_idx], X_num.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]

        if optimize_for is not None:
            threshold, _, _, _ = find_optimal_threshold(
                y_val, y_proba, optimize_for=optimize_for, recall_min=recall_min
            )
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_val)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        fold_metrics.append({
            "ROC_AUC": float(roc_auc_score(y_val, y_proba)),
            "F2": float(fbeta_score(y_val, y_pred, beta=2, zero_division=0)),
            "Recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "Specificity": specificity,
        })

    result = {}
    for metric in ["ROC_AUC", "F2", "Recall", "Precision", "Specificity"]:
        vals = [f[metric] for f in fold_metrics]
        result[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
        result[f"{metric}_std"] = round(float(np.std(vals)), 4)

    logger.info(
        f"CV {n_folds}-fold: ROC_AUC={result['ROC_AUC_mean']:.3f}±{result['ROC_AUC_std']:.3f} | "
        f"F2={result['F2_mean']:.3f}±{result['F2_std']:.3f}"
    )
    return result
