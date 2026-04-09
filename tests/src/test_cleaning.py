# tests/src/test_cleaning.py
import pandas as pd
import pytest
from pathlib import Path
from src.cleaning.cleaner import clean_preop
from src.cleaning.report import generate_cleaning_report


@pytest.fixture
def sample_preop():
    return pd.DataFrame({
        "CODIGO": ["C001", "C002", "C003", "C004"],
        "Documento PMD": ["A001", "A002", "A003", "A004"],
        "Edad": [15, 25, 45, 200],   # 15 < 18, 200 > 120 → outlier
        "Peso": [60.0, 70.0, None, 80.0],
    })


@pytest.fixture
def cleaning_cfg():
    return {
        "identifier_columns": ["CODIGO"],
        "leakage_columns": [],
        "age_filter": {"column": "Edad", "min": 18},
        "outlier_rules": {"Edad": {"min": 0, "max": 120}},
        "encoding_fixes": {},
        "external_services": {},
    }


def test_clean_preop_removes_underage(sample_preop, cleaning_cfg):
    df_clean = clean_preop(sample_preop, cleaning_cfg)
    # NaN values (from outlier marking) are allowed; only known underage values must be absent
    assert (df_clean["Edad"].dropna() >= 18).all()


def test_clean_preop_marks_outliers_as_null(sample_preop, cleaning_cfg):
    df_clean = clean_preop(sample_preop, cleaning_cfg)
    # Edad=200 debe ser nulo después de la limpieza
    assert df_clean["Edad"].isna().sum() >= 1


def test_clean_preop_drops_identifier_columns(sample_preop, cleaning_cfg):
    df_clean = clean_preop(sample_preop, cleaning_cfg)
    assert "CODIGO" not in df_clean.columns


def test_clean_preop_preserves_join_key(sample_preop, cleaning_cfg):
    df_clean = clean_preop(sample_preop, cleaning_cfg)
    assert "Documento PMD" in df_clean.columns


def test_generate_cleaning_report_structure(sample_preop, cleaning_cfg, tmp_path):
    df_clean = clean_preop(sample_preop, cleaning_cfg)
    report = generate_cleaning_report(sample_preop, df_clean, tmp_path)
    assert "rows_before" in report
    assert "rows_after" in report
    assert "cols_dropped" in report
    assert (tmp_path / "cleaning_report.json").exists()
