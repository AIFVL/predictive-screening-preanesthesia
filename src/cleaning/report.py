from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger("cleaning.report")


def generate_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    out_dir: Path | str,
) -> dict:
    """
    Genera estadísticas de limpieza antes/después y las guarda en cleaning_report.json.
    """
    out_dir = Path(out_dir)

    dropped_cols = [c for c in df_before.columns if c not in df_after.columns]
    null_before = round(float(df_before.isnull().mean().mean()) * 100, 2)
    null_after = round(float(df_after.isnull().mean().mean()) * 100, 2)

    report = {
        "rows_before": int(df_before.shape[0]),
        "rows_after": int(df_after.shape[0]),
        "rows_removed": int(df_before.shape[0] - df_after.shape[0]),
        "cols_before": int(df_before.shape[1]),
        "cols_after": int(df_after.shape[1]),
        "cols_dropped": dropped_cols,
        "null_pct_before": null_before,
        "null_pct_after": null_after,
    }

    write_json(report, out_dir / "cleaning_report.json")
    logger.info(
        f"Limpieza: {report['rows_before']:,} → {report['rows_after']:,} filas, "
        f"nulos {null_before:.1f}% → {null_after:.1f}%"
    )
    return report
