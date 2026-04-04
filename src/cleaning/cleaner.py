from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("cleaning.cleaner")


def clean_preop(df: pd.DataFrame, cleaning_cfg: dict) -> pd.DataFrame:
    """
    Aplica las reglas de cleaning_config.yaml al DataFrame preoperatorio.
    Retorna DataFrame limpio (no modifica el original).
    """
    df = df.copy()

    # 1. Aplicar correcciones de encoding en nombres de columnas
    encoding_fixes = cleaning_cfg.get("encoding_fixes", {})
    if encoding_fixes:
        df = df.rename(columns=encoding_fixes)

    # 2. Marcar outliers como nulos (antes de filtrar por edad)
    outlier_rules = cleaning_cfg.get("outlier_rules", {})
    for col, bounds in outlier_rules.items():
        if col not in df.columns:
            continue
        col_min = bounds.get("min")
        col_max = bounds.get("max")
        if col_min is not None:
            df.loc[df[col] < col_min, col] = None
        if col_max is not None:
            df.loc[df[col] > col_max, col] = None

    # 3. Filtro de edad mínima — eliminar filas con edad < min (NaN pasa el filtro)
    age_filter = cleaning_cfg.get("age_filter", {})
    age_col = age_filter.get("column", "Edad")
    age_min = age_filter.get("min")
    if age_min is not None and age_col in df.columns:
        before = len(df)
        df = df[~(df[age_col] < age_min)].reset_index(drop=True)
        logger.info(f"Filtro edad ≥{age_min}: {before} → {len(df)} filas")

    # 4. Eliminar columnas de fuga
    leakage_cols = [c for c in cleaning_cfg.get("leakage_columns", []) if c in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        logger.info(f"Columnas de fuga eliminadas: {leakage_cols}")

    # 5. Eliminar columnas identificadoras
    id_cols = [c for c in cleaning_cfg.get("identifier_columns", []) if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        logger.info(f"Columnas identificadoras eliminadas: {id_cols}")

    return df
