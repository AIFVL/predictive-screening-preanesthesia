# tests/src/test_target.py
import pandas as pd
import pytest
from src.target.pipeline import run_target_extraction


@pytest.fixture
def sample_posop():
    """DataFrame posoperatorio mínimo con columnas que los validadores esperan."""
    return pd.DataFrame({
        "Documento PMD (valoración preanestésica)": ["A001", "A002", "A003"],
        "flag_cancelacion": [0, 1, 0],
        "flag_via_aerea": [0, 0, 1],
        "flag_estancia": [1, 0, 0],
        "flag_reservas": [0, 0, 0],
        "flag_fisiologicas": [0, 0, 0],
        "flag_tiempos": [0, 1, 0],
        "flag_induccion": [0, 0, 0],
        "flag_ventilacion": [0, 0, 0],
        "flag_tecnica": [0, 0, 0],
        "flag_liquidos": [0, 0, 0],
        "flag_desenlace": [0, 0, 0],
        "flag_complicaciones_medicas": [0, 0, 0],
        "flag_seguimiento": [0, 0, 0],
    })


@pytest.fixture
def target_cfg_a():
    return {
        "threshold": 1,
        "apply_cancel_non_medico_rule": True,
        "flags": ["flag_cancelacion", "flag_via_aerea", "flag_estancia"],
    }


def test_run_target_extraction_returns_dataframe(sample_posop, target_cfg_a):
    df_result = run_target_extraction(sample_posop, target_cfg_a)
    assert isinstance(df_result, pd.DataFrame)


def test_run_target_extraction_has_target_column(sample_posop, target_cfg_a):
    df_result = run_target_extraction(sample_posop, target_cfg_a)
    assert "target" in df_result.columns


def test_run_target_extraction_binary_target(sample_posop, target_cfg_a):
    df_result = run_target_extraction(sample_posop, target_cfg_a)
    assert set(df_result["target"].unique()).issubset({0, 1})


def test_run_target_extraction_threshold_1(sample_posop, target_cfg_a):
    """Con threshold=1, cualquier flag activo = target=1."""
    df_result = run_target_extraction(sample_posop, target_cfg_a)
    # A001: flag_estancia=1 → target=1
    # A002: flag_cancelacion=1 → target=1
    # A003: flag_via_aerea=1 → target=1
    assert df_result["target"].sum() == 3


def test_run_target_extraction_threshold_2(sample_posop):
    cfg = {
        "threshold": 2,
        "apply_cancel_non_medico_rule": False,
        "flags": ["flag_cancelacion", "flag_via_aerea", "flag_estancia", "flag_tiempos"],
    }
    df_result = run_target_extraction(sample_posop, cfg)
    # Solo A002 tiene 2 flags (flag_cancelacion + flag_tiempos)
    assert df_result["target"].sum() == 1
