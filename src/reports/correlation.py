from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils.io import write_json
from src.utils.logger import get_logger

logger = get_logger("reports.correlation")


def analyze_preop_posop_correlation(
    df_merged: pd.DataFrame,
    target_name: str,
    out_dir: Path | str,
    min_age: int = 18,
    random_state: int = 42,
) -> dict:
    """
    Calcula Mutual Information entre variables preoperatorias y el target.
    Guarda plot y reporte JSON.
    """
    out_dir = Path(out_dir) / f"eda_correlation_{target_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df_merged.copy()
    if "Edad" in df.columns:
        df = df[df["Edad"] >= min_age].reset_index(drop=True)

    if "target" not in df.columns:
        logger.warning(f"[{target_name}] No hay columna 'target'")
        return {}

    # Seleccionar solo columnas numéricas que no sean flags ni identificadores
    exclude = {"target", "Documento PMD", "Documento_PMD", "n_flags_relevant"}
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude and not c.startswith("flag_")
    ]

    if not numeric_cols:
        logger.warning(f"[{target_name}] Sin columnas numéricas para MI")
        return {}

    X = df[numeric_cols].fillna(-1)
    y = df["target"].astype(int)

    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    mi_df = pd.DataFrame({"feature": numeric_cols, "MI": mi})
    mi_df = mi_df.sort_values("MI", ascending=False).reset_index(drop=True)

    # Plot top 20
    top = mi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["MI"][::-1], color="steelblue", alpha=0.8)
    ax.set_xlabel("Mutual Information")
    ax.set_title(f"Correlación Preop→Target ({target_name}) — Top 20 Features")
    plt.tight_layout()
    plot_path = out_dir / "mi_correlation.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot guardado: {plot_path.name}")

    result = {
        "target": target_name,
        "n_features": len(numeric_cols),
        "top_features": mi_df.head(10).to_dict(orient="records"),
    }
    write_json(result, out_dir / "correlation_report.json")
    logger.info(
        f"[{target_name}] MI analysis completo. "
        f"Top feature: {mi_df.iloc[0]['feature']} (MI={mi_df.iloc[0]['MI']:.4f})"
    )

    return result
