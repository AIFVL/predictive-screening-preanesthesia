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

_BOOTSTRAP_METRICS = ("ROC_AUC", "F2", "Recall", "Precision")


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


def bootstrap_confidence_intervals(
    y_true,
    y_proba,
    threshold: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Estima intervalos de confianza (1-alpha) para ROC_AUC, F2, Recall y Precision
    mediante bootstrap percentil sobre el conjunto de prueba.

    Retorna dict: {metric: {"mean": ..., "ci_lower": ..., "ci_upper": ...}}
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    n = len(y_true_arr)

    samples: dict[str, list[float]] = {m: [] for m in _BOOTSTRAP_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_proba_arr[idx]

        if len(np.unique(yt)) < 2:
            continue

        yhat = (yp >= threshold).astype(int)
        samples["ROC_AUC"].append(float(roc_auc_score(yt, yp)))
        samples["F2"].append(float(fbeta_score(yt, yhat, beta=2, zero_division=0)))
        samples["Recall"].append(float(recall_score(yt, yhat, zero_division=0)))
        samples["Precision"].append(float(precision_score(yt, yhat, zero_division=0)))

    lo, hi = alpha / 2, 1 - alpha / 2
    result: dict[str, dict[str, float]] = {}
    for metric, vals in samples.items():
        arr = np.array(vals)
        result[metric] = {
            "mean": float(arr.mean()),
            "ci_lower": float(np.quantile(arr, lo)),
            "ci_upper": float(np.quantile(arr, hi)),
            "n_valid_samples": len(vals),
        }
        logger.debug(
            f"Bootstrap {metric}: {result[metric]['mean']:.3f} "
            f"[{result[metric]['ci_lower']:.3f}, {result[metric]['ci_upper']:.3f}]"
        )
    return result


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
