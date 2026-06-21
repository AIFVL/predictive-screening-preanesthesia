# tests/src/test_bootstrap_ci.py
import numpy as np
import pytest

from src.evaluation.metrics import bootstrap_confidence_intervals, _BOOTSTRAP_METRICS


@pytest.fixture
def separable_data():
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 100 + [0] * 100)
    y_proba = np.concatenate([
        np.clip(rng.normal(0.75, 0.1, 100), 0.01, 0.99),
        np.clip(rng.normal(0.25, 0.1, 100), 0.01, 0.99),
    ])
    return y_true, y_proba


def test_returns_all_expected_metrics(separable_data):
    y_true, y_proba = separable_data
    result = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5)
    for metric in _BOOTSTRAP_METRICS:
        assert metric in result, f"Falta métrica: {metric}"


def test_each_metric_has_required_keys(separable_data):
    y_true, y_proba = separable_data
    result = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5)
    for metric, values in result.items():
        for key in ("mean", "ci_lower", "ci_upper", "n_valid_samples"):
            assert key in values, f"{metric} falta clave '{key}'"


def test_ci_lower_leq_mean_leq_upper(separable_data):
    y_true, y_proba = separable_data
    result = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5)
    for metric, values in result.items():
        assert values["ci_lower"] <= values["mean"] <= values["ci_upper"], (
            f"{metric}: ci_lower={values['ci_lower']:.3f} > mean={values['mean']:.3f} "
            f"o mean > ci_upper={values['ci_upper']:.3f}"
        )


def test_all_values_in_unit_interval(separable_data):
    y_true, y_proba = separable_data
    result = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5)
    for metric, values in result.items():
        for key in ("mean", "ci_lower", "ci_upper"):
            assert 0.0 <= values[key] <= 1.0, (
                f"{metric}[{key}] fuera de [0,1]: {values[key]}"
            )


def test_deterministic_with_fixed_seed(separable_data):
    y_true, y_proba = separable_data
    r1 = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5, seed=99)
    r2 = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5, seed=99)
    for metric in _BOOTSTRAP_METRICS:
        assert r1[metric]["mean"] == r2[metric]["mean"]
        assert r1[metric]["ci_lower"] == r2[metric]["ci_lower"]
        assert r1[metric]["ci_upper"] == r2[metric]["ci_upper"]


def test_wider_ci_with_fewer_bootstrap_samples(separable_data):
    y_true, y_proba = separable_data
    r_small = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5, n_bootstrap=50, seed=0)
    r_large = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5, n_bootstrap=1000, seed=0)
    width_small = r_small["ROC_AUC"]["ci_upper"] - r_small["ROC_AUC"]["ci_lower"]
    width_large = r_large["ROC_AUC"]["ci_upper"] - r_large["ROC_AUC"]["ci_lower"]
    assert width_small >= width_large


def test_n_valid_samples_matches_n_bootstrap(separable_data):
    y_true, y_proba = separable_data
    n = 200
    result = bootstrap_confidence_intervals(y_true, y_proba, threshold=0.5, n_bootstrap=n)
    for metric, values in result.items():
        assert values["n_valid_samples"] == n, (
            f"{metric}: esperaba {n} muestras, got {values['n_valid_samples']}"
        )
