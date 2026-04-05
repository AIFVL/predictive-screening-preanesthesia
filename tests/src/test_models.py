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


from src.models.registry import build_model
from src.models.trainer import train_model, load_model


@pytest.fixture
def simple_dataset():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "feat_a": rng.randn(100),
        "feat_b": rng.randn(100),
        "feat_c": rng.randn(100),
    })
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


def test_train_model_returns_artifact(simple_dataset, tmp_path):
    X, y = simple_dataset
    cfg = {"module": "sklearn.linear_model", "class": "LogisticRegression",
           "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 200}}
    artifact = train_model(cfg, X, y, out_dir=tmp_path, model_name="lr", random_state=42)
    assert "metrics" in artifact
    assert "threshold" in artifact
    assert "model_path" in artifact


def test_train_model_writes_joblib(simple_dataset, tmp_path):
    X, y = simple_dataset
    cfg = {"module": "sklearn.linear_model", "class": "LogisticRegression",
           "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 200}}
    train_model(cfg, X, y, out_dir=tmp_path, model_name="lr", random_state=42)
    assert (tmp_path / "lr_model.joblib").exists()


def test_train_model_writes_metrics_json(simple_dataset, tmp_path):
    X, y = simple_dataset
    cfg = {"module": "sklearn.linear_model", "class": "LogisticRegression",
           "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 200}}
    train_model(cfg, X, y, out_dir=tmp_path, model_name="lr", random_state=42)
    assert (tmp_path / "lr_metrics.json").exists()


def test_load_model_returns_fitted_model(simple_dataset, tmp_path):
    X, y = simple_dataset
    cfg = {"module": "sklearn.linear_model", "class": "LogisticRegression",
           "params": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 200}}
    artifact = train_model(cfg, X, y, out_dir=tmp_path, model_name="lr", random_state=42)
    model = load_model(artifact["model_path"])
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_build_model_stacking():
    """StackingClassifier se construye correctamente desde config."""
    base_lr = {"module": "sklearn.linear_model", "class": "LogisticRegression",
               "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 200}}
    base_rf = {"module": "sklearn.ensemble", "class": "RandomForestClassifier",
               "params": {"n_estimators": 10}}
    stacking_cfg = {
        "module": "sklearn.ensemble",
        "class": "StackingClassifier",
        "estimators": ["rf", "lr"],
        "meta_estimator": "lr",
        "params": {"cv": 3, "passthrough": False},
        "_estimator_configs": {"rf": base_rf, "lr": base_lr},
    }
    model = build_model(stacking_cfg, random_state=42)
    from sklearn.ensemble import StackingClassifier
    assert isinstance(model, StackingClassifier)
    assert len(model.estimators) == 2


def test_build_model_voting():
    """VotingClassifier soft se construye correctamente."""
    base_lr = {"module": "sklearn.linear_model", "class": "LogisticRegression",
               "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 200}}
    base_rf = {"module": "sklearn.ensemble", "class": "RandomForestClassifier",
               "params": {"n_estimators": 10}}
    voting_cfg = {
        "module": "sklearn.ensemble",
        "class": "VotingClassifier",
        "estimators": ["rf", "lr"],
        "params": {"voting": "soft"},
        "_estimator_configs": {"rf": base_rf, "lr": base_lr},
    }
    model = build_model(voting_cfg, random_state=42)
    from sklearn.ensemble import VotingClassifier
    assert isinstance(model, VotingClassifier)
    assert model.voting == "soft"


def test_build_model_with_calibration():
    """build_model con calibrate=True devuelve CalibratedClassifierCV."""
    cfg = {"module": "sklearn.ensemble", "class": "RandomForestClassifier",
           "params": {"n_estimators": 10}, "calibrate": True}
    model = build_model(cfg, random_state=42)
    from sklearn.calibration import CalibratedClassifierCV
    assert isinstance(model, CalibratedClassifierCV)
