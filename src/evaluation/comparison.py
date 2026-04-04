# src/evaluation/comparison.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import read_json, write_json
from src.utils.logger import get_logger

logger = get_logger("evaluation.comparison")


def aggregate_model_results(
    results_dir: Path | str,
    out_dir: Path | str,
) -> pd.DataFrame:
    """
    Agrega todas las métricas de modelos bajo results_dir.

    Estructura esperada:
        results_dir/<target>/<model>_metrics.json

    Escribe comparison_table.json y retorna DataFrame ordenado por F2 desc.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_file in sorted(results_dir.rglob("*_metrics.json")):
        target = metrics_file.parent.name
        model_name = metrics_file.stem.replace("_metrics", "")
        try:
            m = read_json(metrics_file)
            rows.append({
                "target": target,
                "model": model_name,
                "F2": m.get("F2", float("nan")),
                "ROC_AUC": m.get("ROC_AUC", float("nan")),
                "Recall": m.get("Recall", float("nan")),
                "Precision": m.get("Precision", float("nan")),
                "Threshold": m.get("Threshold", float("nan")),
            })
        except Exception as e:
            logger.warning(f"Error leyendo {metrics_file}: {e}")

    if not rows:
        logger.warning("No se encontraron archivos de métricas")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["target", "F2"], ascending=[True, False])
    write_json(df.to_dict(orient="records"), out_dir / "comparison_table.json")
    logger.info(f"Tabla comparativa: {len(df)} modelos × targets")
    return df
