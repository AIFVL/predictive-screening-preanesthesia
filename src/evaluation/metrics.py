# src/evaluation/metrics.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
)

from src.utils.logger import get_logger

logger = get_logger("evaluation.metrics")


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba,
    threshold: float,
) -> dict:
    """
    Calcula métricas completas de clasificación binaria.
    Retorna dict con: Accuracy, Recall, Precision, F1, F2, ROC_AUC, PR_AUC,
    Balanced_Accuracy, Specificity, Brier, Predicted_Positive_Rate, FN_Rate, Threshold.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fn_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
        "PR_AUC": float(average_precision_score(y_true, y_proba)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Specificity": specificity,
        "Brier": float(brier_score_loss(y_true, y_proba)),
        "Predicted_Positive_Rate": float(np.mean(y_pred)),
        "FN_Rate": fn_rate,
        "Threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true,
    y_proba,
    metric: str = "f2",
    optimize_for: str | None = None,
    recall_min: float = 0.80,
) -> tuple[float, float, list[float], list[float]]:
    """
    Encuentra el umbral de decisión óptimo.

    Modos:
        optimize_for="recall_constraint": garantiza Recall >= recall_min,
            maximiza Precision. Modo clínico recomendado.
        metric="f2"|"f1"|"balanced_accuracy": maximiza la métrica indicada.

    Retorna: (threshold, best_score, all_thresholds, all_scores)
    """
    thresholds = list(np.arange(0.05, 0.95, 0.01))
    scores: list[float] = []
    valid_thresholds: list[float] = []

    if optimize_for == "recall_constraint":
        for t in thresholds:
            y_pred = (np.asarray(y_proba) >= t).astype(int)
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            if rec >= recall_min:
                prec = float(precision_score(y_true, y_pred, zero_division=0))
                valid_thresholds.append(t)
                scores.append(prec)
        if not scores:
            for t in thresholds:
                y_pred = (np.asarray(y_proba) >= t).astype(int)
                valid_thresholds.append(t)
                scores.append(float(recall_score(y_true, y_pred, zero_division=0)))
        best_idx = int(np.argmax(scores))
        return valid_thresholds[best_idx], scores[best_idx], valid_thresholds, scores

    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "f2":
            score = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, y_pred))
        else:
            raise ValueError(f"metric debe ser 'f1', 'f2' o 'balanced_accuracy', recibido: {metric}")
        valid_thresholds.append(t)
        scores.append(score)

    if not scores:
        return 0.5, 0.0, [0.5], [0.0]

    best_idx = int(np.argmax(scores))
    return valid_thresholds[best_idx], scores[best_idx], valid_thresholds, scores
