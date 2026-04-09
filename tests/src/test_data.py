# tests/src/test_data.py
import pandas as pd
import pytest
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.validator import validate_raw_data


@pytest.fixture
def raw_dir(tmp_path):
    """Crea xlsx mínimos simulados."""
    df_pre = pd.DataFrame({
        "Documento PMD": ["A001", "A002", "A003"],
        "Edad": [25, 45, 60],
        "Peso": [70.0, 80.0, 55.0],
    })
    df_pos = pd.DataFrame({
        "Documento PMD (valoración preanestésica)": ["A001", "A002"],
        "target": [1, 0],
    })
    df_pre.to_excel(tmp_path / "OPERA_PRE.xlsx", index=False)
    df_pos.to_excel(tmp_path / "OPERA_POS.xlsx", index=False)
    return tmp_path


def test_load_raw_data_returns_dataframes(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    df_pre, df_pos = load_raw_data(raw_dir, out_dir)
    assert isinstance(df_pre, pd.DataFrame)
    assert isinstance(df_pos, pd.DataFrame)


def test_load_raw_data_writes_parquet(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    load_raw_data(raw_dir, out_dir)
    assert (out_dir / "preop_raw.parquet").exists()
    assert (out_dir / "posop_raw.parquet").exists()


def test_load_raw_data_preserves_rows(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    df_pre, df_pos = load_raw_data(raw_dir, out_dir)
    assert len(df_pre) == 3
    assert len(df_pos) == 2


def test_load_raw_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nonexistent", tmp_path / "out")


def test_load_raw_data_uses_parquet_cache(raw_dir, tmp_path, mocker):
    """Si ya existen los parquet, no releer el xlsx."""
    out_dir = tmp_path / "processed"
    load_raw_data(raw_dir, out_dir)  # Primera carga — crea parquets
    spy = mocker.patch("pandas.read_excel", wraps=pd.read_excel)
    load_raw_data(raw_dir, out_dir)  # Segunda carga — debe usar parquet
    spy.assert_not_called()


def test_validate_raw_data_passes_valid(raw_dir, tmp_path):
    out_dir = tmp_path / "processed"
    df_pre, df_pos = load_raw_data(raw_dir, out_dir)
    report = validate_raw_data(df_pre, df_pos, out_dir)
    assert report["status"] == "ok"
    assert "preop" in report
    assert "posop" in report


def test_validate_raw_data_writes_report(raw_dir, tmp_path):
    import json
    out_dir = tmp_path / "processed"
    df_pre, df_pos = load_raw_data(raw_dir, out_dir)
    validate_raw_data(df_pre, df_pos, out_dir)
    report_path = out_dir / "validation_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "preop" in report


def test_validate_warns_on_missing_join_key(tmp_path):
    # DataFrame sin columna de join
    df_pre = pd.DataFrame({"Edad": [25, 45]})
    df_pos = pd.DataFrame({"target": [1, 0]})
    out_dir = tmp_path / "processed"
    report = validate_raw_data(df_pre, df_pos, out_dir)
    assert report["status"] == "warnings"
    assert any("Documento PMD" in w for w in report["warnings"])
