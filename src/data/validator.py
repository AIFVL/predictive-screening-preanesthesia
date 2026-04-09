from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger("data.validator")

JOIN_KEY_PRE = "Documento PMD"
JOIN_KEY_POS_VARIANTS = ["Documento PMD", "Documento PMD (valoración preanestésica)"]


def validate_raw_data(
    df_pre: pd.DataFrame,
    df_pos: pd.DataFrame,
    out_dir: Path | str,
) -> dict:
    """
    Valida esquema básico y completitud de los DataFrames crudos.
    Escribe validation_report.json en out_dir.
    Retorna dict con status ('ok' | 'warnings'), warnings list, y stats.
    """
    out_dir = Path(out_dir)
    warnings: list[str] = []

    # Verificar clave de join en preop
    if JOIN_KEY_PRE not in df_pre.columns:
        warnings.append(f"Preop: columna '{JOIN_KEY_PRE}' no encontrada")

    # Verificar que posop tiene alguna variante de la clave de join
    pos_has_key = any(k in df_pos.columns for k in JOIN_KEY_POS_VARIANTS)
    if not pos_has_key:
        warnings.append(f"Posop: ninguna variante de 'Documento PMD' encontrada")

    # Verificar columna target en posop
    if "target" not in df_pos.columns:
        warnings.append("Posop: columna 'target' no encontrada")

    def _col_stats(df: pd.DataFrame) -> dict:
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "null_pct": round(float(df.isnull().mean().mean()) * 100, 2),
            "columns": list(df.columns),
        }

    report = {
        "status": "ok" if not warnings else "warnings",
        "warnings": warnings,
        "preop": _col_stats(df_pre),
        "posop": _col_stats(df_pos),
    }

    write_json(report, out_dir / "validation_report.json")
    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Validación OK — sin advertencias")

    return report
