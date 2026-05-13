# src/evaluation/shap_groups.py
"""
Análisis comparativo de SHAP values por grupo clínico (FN / TP / FP / TN).
Lee los archivos .npy ya generados por run_shap_plots — no recalcula SHAP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("evaluation.shap_groups")

_CASE_TYPES = ["TP", "FN", "FP", "TN"]


def load_shap_artifacts(shap_dir: Path, model_key: str) -> tuple[np.ndarray, list[str], float]:
    """
    Carga shap_values, feature_names y expected_value desde disco.
    Lanza FileNotFoundError si el modelo no tiene SHAP generado.
    """
    sv_path   = shap_dir / f"shap_values_{model_key}.npy"
    feat_path = shap_dir / f"shap_features_{model_key}.txt"
    ev_path   = shap_dir / f"shap_expected_{model_key}.txt"

    for p in (sv_path, feat_path, ev_path):
        if not p.exists():
            raise FileNotFoundError(
                f"SHAP artifact no encontrado: {p}. "
                f"Ejecuta primero run_shap_plots para {model_key}."
            )

    shap_values = np.load(sv_path)
    # El archivo puede estar en utf-8 (nuevas ejecuciones) o cp1252 (Windows legacy)
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            feature_names = feat_path.read_text(encoding=enc).splitlines()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        feature_names = feat_path.read_bytes().decode("latin-1").splitlines()
    expected_value = float(ev_path.read_text(encoding="utf-8", errors="replace").strip())
    return shap_values, feature_names, expected_value


def compute_shap_group_profiles(
    shap_values: np.ndarray,
    feature_names: list[str],
    cases_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Calcula el SHAP medio por grupo clínico (TP / FN / FP / TN).

    Parámetros
    ----------
    shap_values   : array (n_test_samples, n_features) — índice posicional del test set.
    feature_names : lista de nombres de features (len = n_features).
    cases_df      : DataFrame con columnas 'case_index' y 'case_type'.
                    case_index es el índice original del DataFrame X_test.
    top_n         : número de features a incluir en el output (ordenadas por |Delta FN-TP|).

    Retorna
    -------
    DataFrame con columnas:
        Feature, SHAP_TP, SHAP_FN, SHAP_FP, SHAP_TN,
        Delta_TP_FN  (TP - FN: positivo = TP tiene más señal),
        AbsDelta     (|Delta_TP_FN| — para ordenar),
        N_TP, N_FN, N_FP, N_TN  (conteos por grupo).
    """
    n_samples = shap_values.shape[0]

    # cases_df.case_index es el índice del DataFrame original de X_test.
    # shap_values tiene índice posicional 0..n_test-1.
    # Necesitamos mapear case_index → posición en el array.
    # Asumimos que case_index es el índice original del split y
    # el array SHAP fue generado en el mismo orden que X_test.parquet.
    # Construimos un mapeo índice_original → posición_array.
    all_case_indices = cases_df["case_index"].values
    all_case_types   = cases_df["case_type"].values

    # El array SHAP cubre TODO el test set; cases_df solo tiene FN/FP/TP/TN
    # seleccionados (max 10 por grupo). Para el perfil necesitamos TODOS los casos.
    # Usamos los índices originales para mapear a posiciones del array.
    # Si case_index es entero y continuo desde 0, posición = case_index mismo.
    # Si no, construimos el mapa completo a partir del parquet (hecho en el caller).
    # Aquí recibimos pos_map si disponible como columna 'pos' en cases_df.
    if "pos" in cases_df.columns:
        positions = cases_df["pos"].values
    else:
        # Fallback: asumir que case_index == posición posicional en el array
        positions = cases_df["case_index"].values.astype(int)

    # Filtrar posiciones válidas (dentro del rango del array)
    valid = (positions >= 0) & (positions < n_samples)
    positions = positions[valid]
    case_types = all_case_types[valid]

    results = {}
    counts  = {}
    for ct in _CASE_TYPES:
        mask = case_types == ct
        idxs = positions[mask]
        counts[ct] = int(mask.sum())
        if len(idxs) > 0:
            results[ct] = shap_values[idxs].mean(axis=0)
        else:
            results[ct] = np.zeros(len(feature_names))
            logger.warning(f"  Grupo {ct} sin casos — SHAP medio = 0 para todas las features.")

    df = pd.DataFrame({
        "Feature":  feature_names,
        "SHAP_TP":  results["TP"],
        "SHAP_FN":  results["FN"],
        "SHAP_FP":  results["FP"],
        "SHAP_TN":  results["TN"],
    })
    df["Delta_TP_FN"] = df["SHAP_TP"] - df["SHAP_FN"]
    df["AbsDelta"]    = df["Delta_TP_FN"].abs()
    df["N_TP"] = counts["TP"]
    df["N_FN"] = counts["FN"]
    df["N_FP"] = counts["FP"]
    df["N_TN"] = counts["TN"]

    df = df.sort_values("AbsDelta", ascending=False).reset_index(drop=True)

    logger.info(
        f"  Grupos: TP={counts['TP']} FN={counts['FN']} "
        f"FP={counts['FP']} TN={counts['TN']}"
    )
    return df.head(top_n).copy()


def build_fn_insight_table(profiles_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Genera una tabla clínica legible centrada en los FN:
    qué features distinguen más a los FN de los TP.

    Columnas:
        Feature, SHAP_TP, SHAP_FN, Delta, Interpretacion
    """
    df = profiles_df.head(top_n).copy()

    def interpret(row) -> str:
        delta = row["Delta_TP_FN"]
        shap_fn = row["SHAP_FN"]
        shap_tp = row["SHAP_TP"]
        if delta > 0:
            # TP tiene más SHAP positivo que FN
            if shap_fn < 0:
                return f"Protege al FN (baja riesgo −{abs(shap_fn):.3f}), eleva al TP (+{shap_tp:.3f})"
            else:
                return f"Señal débil en FN (+{shap_fn:.3f}) vs fuerte en TP (+{shap_tp:.3f})"
        else:
            if shap_tp < 0:
                return f"Protege más al TP que al FN — FN tiene más riesgo aquí"
            else:
                return f"FN tiene mayor señal positiva (+{shap_fn:.3f}) — el modelo la ve pero no alcanza"

    df["Interpretacion"] = df.apply(interpret, axis=1)
    return df[["Feature", "SHAP_TP", "SHAP_FN", "Delta_TP_FN", "Interpretacion"]].copy()
