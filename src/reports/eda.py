from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Sin GUI — necesario en Docker/scripts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("reports.eda")


def _save_and_close(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot guardado: {path.name}")


def generate_preop_eda(df: pd.DataFrame, out_dir: Path | str, label: str = "preop") -> None:
    """
    Genera plots exploratorios básicos del DataFrame preoperatorio.
    Guarda en out_dir/<label>/.
    """
    out_dir = Path(out_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 1. Distribución de variables numéricas
    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = [axes] if n_rows * n_cols == 1 else list(axes.flat)
        for ax, col in zip(axes_flat, numeric_cols):
            df[col].dropna().hist(bins=30, ax=ax, color="steelblue", alpha=0.7)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
        for ax in axes_flat[len(numeric_cols):]:
            ax.set_visible(False)
        fig.suptitle(f"EDA {label} — Distribuciones Numéricas", fontsize=12)
        plt.tight_layout()
        _save_and_close(fig, out_dir / "numeric_distributions.png")

    # 2. Heatmap de nulos
    null_pct = df.isnull().mean().sort_values(ascending=False)
    cols_with_nulls = null_pct[null_pct > 0]
    if not cols_with_nulls.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, len(cols_with_nulls) * 0.3)))
        cols_with_nulls.plot.barh(ax=ax, color="tomato", alpha=0.8)
        ax.set_xlabel("Proporción nulos")
        ax.set_title(f"EDA {label} — Porcentaje de Nulos por Columna")
        plt.tight_layout()
        _save_and_close(fig, out_dir / "nulls_by_column.png")


def generate_posop_eda(
    df_merged: pd.DataFrame,
    out_dir: Path | str,
    target_name: str,
) -> None:
    """
    Genera EDA del dataset mergeado (preop + target).
    Incluye distribución del target y prevalencia.
    """
    out_dir = Path(out_dir) / f"eda_posop_{target_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "target" not in df_merged.columns:
        logger.warning("No hay columna 'target' — omitiendo EDA posop")
        return

    # 1. Distribución del target
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_merged["target"].value_counts().sort_index()
    counts.plot.bar(ax=ax, color=["steelblue", "tomato"], alpha=0.8)
    prevalence = df_merged["target"].mean() * 100
    ax.set_title(f"{target_name} — Target (prevalencia={prevalence:.1f}%)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Casos")
    ax.set_xticklabels(["Negativo (0)", "Positivo (1)"], rotation=0)
    plt.tight_layout()
    _save_and_close(fig, out_dir / "target_distribution.png")
