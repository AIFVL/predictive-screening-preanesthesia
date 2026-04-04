from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet
from src.utils.logger import get_logger

logger = get_logger("data.loader")


def load_raw_data(
    raw_dir: Path | str,
    out_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga OPERA_PRE.xlsx y OPERA_POS.xlsx desde raw_dir.
    Escribe parquets en out_dir y los reutiliza en llamadas posteriores.
    Retorna (df_preop, df_posop).
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_parquet = out_dir / "preop_raw.parquet"
    pos_parquet = out_dir / "posop_raw.parquet"

    if pre_parquet.exists() and pos_parquet.exists():
        logger.info("Usando parquet cacheados — omitiendo lectura de xlsx")
        return read_parquet(pre_parquet), read_parquet(pos_parquet)

    pre_xlsx = raw_dir / "OPERA_PRE.xlsx"
    pos_xlsx = raw_dir / "OPERA_POS.xlsx"

    if not pre_xlsx.exists():
        raise FileNotFoundError(f"No se encontró: {pre_xlsx}")
    if not pos_xlsx.exists():
        raise FileNotFoundError(f"No se encontró: {pos_xlsx}")

    logger.info(f"Leyendo {pre_xlsx.name} ...")
    df_pre = pd.read_excel(pre_xlsx)
    logger.info(f"  → {df_pre.shape[0]:,} filas × {df_pre.shape[1]} columnas")

    logger.info(f"Leyendo {pos_xlsx.name} ...")
    df_pos = pd.read_excel(pos_xlsx)
    logger.info(f"  → {df_pos.shape[0]:,} filas × {df_pos.shape[1]} columnas")

    write_parquet(df_pre, pre_parquet)
    write_parquet(df_pos, pos_parquet)
    logger.info(f"Parquets guardados en {out_dir}")

    return df_pre, df_pos
