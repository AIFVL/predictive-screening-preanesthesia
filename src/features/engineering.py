from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("features.engineering")

# Fallback — los valores reales vienen de features_config.yaml via PipelineConfig
_DEFAULT_ENCODING_FIX_MAP: dict[str, str] = {
    "PrÃ³tesis Dental_movil": "Prótesis Dental_movil",
    "PrÃ³tesis Dental_no": "Prótesis Dental_no",
    "AlÃ©rgeno_med_opioides": "Alérgeno_med_opioides",
    "AlÃ©rgeno_suplementos": "Alérgeno_suplementos",
    "TensiÃ³n Arterial SistÃ³lica (mm/Hg)": "Tensión Arterial Sistólica (mm/Hg)",
    "TensiÃ³n Arterial DiastÃ³lica (mm/Hg)": "Tensión Arterial Diastólica (mm/Hg)",
    "TensiÃ³n Arterial Media (mm/Hg)": "Tensión Arterial Media (mm/Hg)",
}


def apply_encoding_fixes(df: pd.DataFrame, encoding_fix_map: dict | None = None) -> pd.DataFrame:
    """Renombra columnas con caracteres de encoding incorrecto."""
    fix_map = encoding_fix_map if encoding_fix_map is not None else _DEFAULT_ENCODING_FIX_MAP
    return df.rename(columns={k: v for k, v in fix_map.items() if k in df.columns})


def infer_candidate_features(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    min_numeric_ratio: float = 0.4,
) -> list[str]:
    """
    Infiere columnas candidatas como features:
    - Excluye columnas de join, target, flags
    - Requiere que al menos min_numeric_ratio de valores sean numéricos
    - Excluye columnas constantes
    """
    excluded = {"target", "Documento PMD", "Documento_PMD", "n_flags_relevant"}
    if exclude_cols:
        excluded.update(exclude_cols)

    candidates = []
    for col in df.columns:
        if col in excluded or col.startswith("flag_"):
            continue
        series_num = pd.to_numeric(df[col], errors="coerce")
        if float(series_num.notna().mean()) < min_numeric_ratio:
            continue
        if series_num.nunique(dropna=True) <= 1:
            continue
        candidates.append(col)

    return candidates


def sanitize_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Elimina features constantes (varianza 0) en el subset actual.
    Retorna (valid_features, dropped_features).
    """
    valid, dropped = [], []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].nunique() <= 1:
            dropped.append(col)
        else:
            valid.append(col)

    if dropped:
        logger.info(f"Sanitización: {len(dropped)} columnas constantes eliminadas")

    return valid, dropped
