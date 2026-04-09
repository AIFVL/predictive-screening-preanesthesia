# tests/src/test_reports_prepost.py
import numpy as np
import pandas as pd
import pytest

from src.reports.pre_post_analysis import (
    _prepare_numeric_pre_features,
    _compute_linkage_for_post_variable,
    run_pre_post_linkage_analysis,
)


@pytest.fixture
def merged_df():
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "Documento PMD": range(n),
        "Edad": rng.randint(18, 80, n),
        "feat_num_a": rng.randn(n),
        "feat_num_b": rng.randn(n),
        "feat_num_c": rng.randn(n),
        "target": (rng.randn(n) > 0).astype(int),
        "flag_complication_a": (rng.randn(n) > 0.5).astype(int),
    })
    return df


def test_prepare_numeric_pre_features_drops_nonnumeric(merged_df):
    df_with_str = merged_df.copy()
    df_with_str["text_col"] = "abc"
    result = _prepare_numeric_pre_features(df_with_str)
    assert "text_col" not in result.columns
    assert "Documento PMD" not in result.columns


def test_prepare_numeric_pre_features_imputes_nan():
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, np.nan, 4.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0],
    })
    result = _prepare_numeric_pre_features(df)
    assert result["feat_a"].isna().sum() == 0


def test_compute_linkage_columns(merged_df):
    X_pre = merged_df[["feat_num_a", "feat_num_b"]].copy()
    X_pre = X_pre.fillna(-1)
    y_post = merged_df["target"]
    result = _compute_linkage_for_post_variable(X_pre, y_post, "target_a", "target", top_n=10, random_state=42)
    assert list(result.columns) == [
        "Version", "Post_Variable", "Feature",
        "Mutual_Information", "Abs_Pearson", "Combined_Score"
    ]


def test_compute_linkage_combined_score_formula(merged_df):
    X_pre = merged_df[["feat_num_a", "feat_num_b"]].fillna(-1)
    y_post = merged_df["target"]
    result = _compute_linkage_for_post_variable(X_pre, y_post, "target_a", "target", top_n=None, random_state=42)
    expected = 0.7 * result["Mutual_Information"] + 0.3 * result["Abs_Pearson"]
    pd.testing.assert_series_equal(result["Combined_Score"].reset_index(drop=True),
                                   expected.reset_index(drop=True), check_names=False)


def test_compute_linkage_empty_when_single_class():
    X_pre = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y_post = pd.Series([1, 1, 1])  # no variance
    result = _compute_linkage_for_post_variable(X_pre, y_post, "v", "flag_x", top_n=5, random_state=42)
    assert result.empty


def test_run_pre_post_linkage_analysis_outputs(merged_df, tmp_path):
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    target_dir = proc_dir / "target_a"
    target_dir.mkdir()
    merged_df.to_parquet(target_dir / "merged.parquet", index=False)

    output_dir = tmp_path / "reports"
    result = run_pre_post_linkage_analysis(
        proc_dir=proc_dir,
        output_dir=output_dir,
        active_targets=["target_a"],
        top_n=5,
        random_state=42,
    )
    assert "per_flag_linkage" in result
    assert "flag_summary" in result
    assert (output_dir / "pre_post_signal" / "pre_post_linkage_per_flag.csv").exists()
    assert (output_dir / "pre_post_signal" / "pre_post_linkage_summary.csv").exists()


def test_run_pre_post_linkage_analysis_summary_columns(merged_df, tmp_path):
    proc_dir = tmp_path / "proc"
    (proc_dir / "target_a").mkdir(parents=True)
    merged_df.to_parquet(proc_dir / "target_a" / "merged.parquet", index=False)

    output_dir = tmp_path / "reports"
    result = run_pre_post_linkage_analysis(
        proc_dir=proc_dir,
        output_dir=output_dir,
        active_targets=["target_a"],
    )
    summary = result["flag_summary"]
    for col in ["Version", "Post_Variable", "Prevalencia", "Max_MI", "Max_Pearson",
                "N_Features_Informativas", "Top3_Features_PRE"]:
        assert col in summary.columns
