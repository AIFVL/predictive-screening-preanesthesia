# tests/src/test_models.py
import numpy as np
import pandas as pd
import pytest
from src.evaluation.metrics import compute_classification_metrics, find_optimal_threshold


@pytest.fixture
def binary_predictions():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.1, 0.85, 0.4, 0.75, 0.05])
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


def test_compute_metrics_keys(binary_predictions):
    y_true, y_pred, y_proba = binary_predictions
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, threshold=0.5)
    for key in ["Recall", "Precision", "F1", "F2", "ROC_AUC", "PR_AUC", "Threshold"]:
        assert key in metrics, f"Falta clave: {key}"


def test_compute_metrics_values_in_range(binary_predictions):
    y_true, y_pred, y_proba = binary_predictions
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, threshold=0.5)
    for key in ["Recall", "Precision", "F1", "F2", "ROC_AUC", "PR_AUC"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} fuera de rango: {metrics[key]}"


def test_find_optimal_threshold_returns_valid(binary_predictions):
    y_true, _, y_proba = binary_predictions
    threshold, score, _, _ = find_optimal_threshold(y_true, y_proba, metric="f2")
    assert 0.0 < threshold < 1.0
    assert score >= 0.0


def test_find_optimal_threshold_recall_constraint(binary_predictions):
    y_true, _, y_proba = binary_predictions
    threshold, _, _, _ = find_optimal_threshold(
        y_true, y_proba, optimize_for="recall_constraint", recall_min=0.8
    )
    from sklearn.metrics import recall_score
    y_pred = (y_proba >= threshold).astype(int)
    assert recall_score(y_true, y_pred) >= 0.75  # margen de tolerancia


def test_compute_metrics_perfect_classifier():
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba = np.array([0.99, 0.01, 0.98, 0.02, 0.97])
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, threshold=0.5)
    assert metrics["Recall"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["ROC_AUC"] == 1.0
