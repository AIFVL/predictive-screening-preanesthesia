import pandas as pd
import pytest
from pathlib import Path
from src.datasets.merge import merge_preop_target
from src.datasets.splits import stratified_train_test_split


@pytest.fixture
def df_preop():
    return pd.DataFrame({
        "Documento PMD": [f"P{i:03d}" for i in range(100)],
        "Edad": [30 + i % 40 for i in range(100)],
        "Peso": [60.0 + i % 30 for i in range(100)],
    })


@pytest.fixture
def df_target():
    return pd.DataFrame({
        "Documento PMD": [f"P{i:03d}" for i in range(80)],  # 80 de 100 hacen match
        "target": [i % 3 == 0 for i in range(80)],  # ~33% positivos
    })


def test_merge_returns_dataframe(df_preop, df_target):
    result = merge_preop_target(df_preop, df_target)
    assert isinstance(result, pd.DataFrame)


def test_merge_has_target_column(df_preop, df_target):
    result = merge_preop_target(df_preop, df_target)
    assert "target" in result.columns


def test_merge_inner_join_count(df_preop, df_target):
    result = merge_preop_target(df_preop, df_target)
    assert len(result) == 80  # inner join — solo los que hacen match


def test_merge_handles_column_alias(df_preop):
    """Posop puede tener columna con alias largo."""
    df_pos = pd.DataFrame({
        "Documento PMD (valoración preanestésica)": [f"P{i:03d}" for i in range(50)],
        "target": [i % 2 for i in range(50)],
    })
    result = merge_preop_target(df_preop, df_pos)
    assert len(result) == 50
    assert "target" in result.columns


def test_split_returns_four_arrays(df_preop, df_target):
    merged = merge_preop_target(df_preop, df_target)
    X = merged.drop(columns=["target", "Documento PMD"])
    y = merged["target"].astype(int)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_writes_parquets(df_preop, df_target, tmp_path):
    merged = merge_preop_target(df_preop, df_target)
    X = merged.drop(columns=["target", "Documento PMD"])
    y = merged["target"].astype(int)
    stratified_train_test_split(X, y, test_size=0.2, out_dir=tmp_path)
    assert (tmp_path / "X_train.parquet").exists()
    assert (tmp_path / "X_test.parquet").exists()
    assert (tmp_path / "y_train.parquet").exists()
    assert (tmp_path / "y_test.parquet").exists()


def test_split_stratified_preserves_ratio(df_preop, df_target):
    merged = merge_preop_target(df_preop, df_target)
    X = merged.drop(columns=["target", "Documento PMD"])
    y = merged["target"].astype(int)
    _, _, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2)
    ratio_full = y.mean()
    ratio_test = y_test.mean()
    assert abs(ratio_full - ratio_test) < 0.1  # dentro del 10%
