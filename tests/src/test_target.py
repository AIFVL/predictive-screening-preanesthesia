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


@pytest.fixture
def sample_posop_subflag():
    """DataFrame con sub-flags ya calculados (como los produce specific/*.py)."""
    return pd.DataFrame({
        "Documento PMD (valoración preanestésica)": ["P1", "P2", "P3", "P4"],
        # sub-flags de via_aerea
        "flag_intubacion_dificil":         [1, 0, 0, 0],
        "flag_tipo_intubacion_complejo":    [0, 0, 0, 0],
        "flag_elemento_via_aerea_complejo": [0, 0, 0, 0],
        # sub-flags de estancia
        "flag_uci_no_planeada":            [0, 1, 0, 0],
        "flag_estancia_uci":               [0, 1, 0, 0],
        # sub-flags de desenlace
        "flag_destino_uci":                [0, 0, 0, 0],
        "flag_intubado_salida":            [0, 1, 0, 0],
        # otros flags
        "flag_complicaciones_medicas":     [0, 0, 1, 0],
        "flag_liquidos":                   [0, 0, 0, 0],
        "flag_seguimiento":                [0, 0, 0, 0],
        "flag_hospitalizacion_no_anticipada": [0, 0, 0, 1],
        # flags agregados para evitar que apply_all_validations los recalcule
        "flag_cancelacion":  [0, 0, 0, 0],
        "flag_via_aerea":    [1, 0, 0, 0],
        "flag_estancia":     [0, 1, 0, 0],
        "flag_desenlace":    [0, 0, 0, 0],
        "flag_reservas":     [0, 0, 0, 0],
        "flag_fisiologicas": [0, 0, 0, 0],
        "flag_tiempos":      [0, 0, 0, 0],
        "flag_induccion":    [0, 0, 0, 0],
        "flag_ventilacion":  [0, 0, 0, 0],
        "flag_tecnica":      [0, 0, 0, 0],
    })


def test_subflag_logic_v2(sample_posop_subflag):
    """D-v2: target from refined sub-flags, no hospitalización."""
    cfg = {
        "threshold": 1,
        "apply_cancel_non_medico_rule": False,
        "flags": ["flag_via_aerea", "flag_estancia", "flag_desenlace",
                  "flag_complicaciones_medicas", "flag_liquidos", "flag_seguimiento"],
        "subflag_logic": {
            "flag_via_aerea_r": "flag_intubacion_dificil OR flag_tipo_intubacion_complejo OR flag_elemento_via_aerea_complejo",
            "flag_estancia_r": "flag_uci_no_planeada OR flag_estancia_uci",
            "flag_desenlace_r": "flag_destino_uci OR (flag_intubado_salida AND flag_uci_no_planeada)",
            "_target_flags": "flag_via_aerea_r OR flag_estancia_r OR flag_desenlace_r OR flag_complicaciones_medicas OR flag_liquidos OR flag_seguimiento",
        },
    }
    df_result = run_target_extraction(sample_posop_subflag, cfg)
    # P1: flag_intubacion_dificil=1 → flag_via_aerea_r=1 → target=1
    # P2: flag_uci_no_planeada=1 → flag_estancia_r=1 AND flag_desenlace_r=1 → target=1
    # P3: flag_complicaciones_medicas=1 → target=1
    # P4: only flag_hospitalizacion_no_anticipada=1, not in _target_flags → target=0
    assert list(df_result["target"]) == [1, 1, 1, 0]


def test_subflag_logic_v2_hosp(sample_posop_subflag):
    """D-v2+hosp: same as v2 but includes hospitalización no anticipada."""
    cfg = {
        "threshold": 1,
        "apply_cancel_non_medico_rule": False,
        "flags": ["flag_via_aerea", "flag_estancia", "flag_desenlace",
                  "flag_complicaciones_medicas", "flag_liquidos", "flag_seguimiento"],
        "subflag_logic": {
            "flag_via_aerea_r": "flag_intubacion_dificil OR flag_tipo_intubacion_complejo OR flag_elemento_via_aerea_complejo",
            "flag_estancia_r": "flag_uci_no_planeada OR flag_estancia_uci",
            "flag_desenlace_r": "flag_destino_uci OR (flag_intubado_salida AND flag_uci_no_planeada)",
            "_target_flags": "flag_via_aerea_r OR flag_estancia_r OR flag_desenlace_r OR flag_complicaciones_medicas OR flag_liquidos OR flag_seguimiento OR flag_hospitalizacion_no_anticipada",
        },
    }
    df_result = run_target_extraction(sample_posop_subflag, cfg)
    # P4: flag_hospitalizacion_no_anticipada=1 → target=1 now
    assert list(df_result["target"]) == [1, 1, 1, 1]
