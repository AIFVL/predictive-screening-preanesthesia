# tests/src/test_evaluation_explainability.py
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.explainability import compute_global_explainability, build_case_review_table


@pytest.fixture
def trained_rf():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "feat_a": rng.randn(200),
        "feat_b": rng.randn(200),
        "feat_c": rng.randn(200),
    })
    y = ((X["feat_a"] + X["feat_b"]) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_lr():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "feat_a": rng.randn(200),
        "feat_b": rng.randn(200),
    })
    y = ((X["feat_a"] + X["feat_b"]) > 0).astype(int)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_explainability_rf_columns(trained_rf):
    model, X, y = trained_rf
    df = compute_global_explainability(model, X, y)
    assert list(df.columns) == ["Feature", "Importance", "Source"]


def test_explainability_rf_source(trained_rf):
    model, X, y = trained_rf
    df = compute_global_explainability(model, X, y)
    assert df["Source"].iloc[0] == "native_feature_importance"


def test_explainability_rf_top_n(trained_rf):
    model, X, y = trained_rf
    df = compute_global_explainability(model, X, y, top_n=2)
    assert len(df) == 2


def test_explainability_rf_sorted_descending(trained_rf):
    model, X, y = trained_rf
    df = compute_global_explainability(model, X, y)
    assert df["Importance"].is_monotonic_decreasing


def test_explainability_lr_uses_abs_coef(trained_lr):
    model, X, y = trained_lr
    df = compute_global_explainability(model, X, y)
    assert df["Source"].iloc[0] == "abs_coef"
    assert (df["Importance"] >= 0).all()


def test_build_case_review_table_columns():
    rng = np.random.RandomState(42)
    idx = list(range(40))
    X_test = pd.DataFrame({"feat_a": rng.randn(40), "feat_b": rng.randn(40)}, index=idx)
    y_test = pd.Series([1, 0] * 20, index=idx)
    y_pred = np.array([1, 0] * 20)
    y_proba = np.clip(rng.rand(40), 0.01, 0.99)
    top_features = ["feat_a", "feat_b"]

    result = build_case_review_table(X_test, y_test, y_pred, y_proba, top_features)
    assert list(result.columns) == ["case_index", "case_type", "y_true", "y_pred", "y_proba", "reason"]


def test_build_case_review_table_case_types():
    rng = np.random.RandomState(0)
    idx = list(range(40))
    X_test = pd.DataFrame({"feat_a": rng.randn(40)}, index=idx)
    y_test = pd.Series([1, 0] * 20, index=idx)
    y_pred = np.array([1, 0, 0, 1] * 10)  # mix of TP/TN/FN/FP
    y_proba = np.clip(rng.rand(40), 0.01, 0.99)

    result = build_case_review_table(X_test, y_test, y_pred, y_proba, ["feat_a"])
    assert set(result["case_type"].unique()).issubset({"TP", "TN", "FN", "FP"})


def test_build_case_review_table_max_per_group():
    rng = np.random.RandomState(1)
    idx = list(range(100))
    X_test = pd.DataFrame({"f": rng.randn(100)}, index=idx)
    y_test = pd.Series([1] * 50 + [0] * 50, index=idx)
    y_pred = np.array([1] * 50 + [0] * 50)
    y_proba = np.linspace(0.6, 0.99, 50).tolist() + np.linspace(0.01, 0.39, 50).tolist()

    result = build_case_review_table(X_test, y_test, y_pred, y_proba, ["f"], max_cases_per_group=5)
    assert (result.groupby("case_type").size() <= 5).all()


def test_build_case_review_table_reason_format():
    idx = [10, 11, 12, 13]
    X_test = pd.DataFrame({"feat_a": [0.5, -0.3, 1.2, 0.0]}, index=idx)
    y_test = pd.Series([1, 0, 1, 0], index=idx)
    y_pred = np.array([1, 0, 0, 1])
    y_proba = np.array([0.8, 0.1, 0.4, 0.7])

    result = build_case_review_table(X_test, y_test, y_pred, y_proba, ["feat_a"])
    for reason in result["reason"]:
        if reason:
            assert "feat_a=" in reason
