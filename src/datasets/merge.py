from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("datasets.merge")

JOIN_KEY = "Documento PMD"
JOIN_KEY_ALIAS = "Documento PMD (valoración preanestésica)"


def merge_preop_target(
    df_preop: pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge inner entre preop y target usando 'Documento PMD' como clave.
    Normaliza el alias largo de la columna en posop si es necesario.
    Retorna DataFrame merged con columna 'target'.
    """
    df_pre = df_preop.copy()
    df_tgt = df_target.copy()

    # Normalizar alias de columna join en posop
    if JOIN_KEY_ALIAS in df_tgt.columns and JOIN_KEY not in df_tgt.columns:
        df_tgt = df_tgt.rename(columns={JOIN_KEY_ALIAS: JOIN_KEY})

    if JOIN_KEY not in df_pre.columns:
        raise ValueError(f"Preop no tiene columna '{JOIN_KEY}'")
    if JOIN_KEY not in df_tgt.columns:
        raise ValueError(f"Posop no tiene columna '{JOIN_KEY}' ni su alias")
    if "target" not in df_tgt.columns:
        raise ValueError("Posop no tiene columna 'target'")

    before_pre = len(df_pre)
    df_merged = pd.merge(
        df_pre,
        df_tgt[[JOIN_KEY, "target"]],
        on=JOIN_KEY,
        how="inner",
    )

    logger.info(
        f"Merge: {before_pre:,} preop × {len(df_tgt):,} posop → "
        f"{len(df_merged):,} filas ({df_merged['target'].mean() * 100:.1f}% positivos)"
    )
    return df_merged
