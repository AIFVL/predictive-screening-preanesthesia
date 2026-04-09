# src/reports/comparison_plots.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("reports.comparison_plots")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot: {path.name}")


def plot_comparison_heatmap(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path | str,
) -> None:
    """
    Heatmap de <metric> con filas=targets y columnas=modelos.
    df debe tener columnas: target, model, <metric>.
    """
    import seaborn as sns
    out_dir = Path(out_dir)
    if df.empty or metric not in df.columns:
        logger.warning(f"DataFrame vacío o métrica '{metric}' no encontrada")
        return

    pivot = df.pivot_table(index="target", columns="model", values=metric)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(4, len(pivot) * 1.2)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(f"{metric} por Target × Modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Target")
    plt.tight_layout()
    _save(fig, out_dir / f"comparison_{metric.lower()}.png")


def plot_ranking_bars(
    df: pd.DataFrame,
    metric: str,
    out_dir: Path | str,
) -> None:
    """
    Barras agrupadas por target para comparar modelos en <metric>.
    """
    out_dir = Path(out_dir)
    if df.empty or metric not in df.columns:
        return

    targets = df["target"].unique()
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, sorted(targets)):
        sub = df[df["target"] == target].sort_values(metric, ascending=False)
        ax.bar(sub["model"], sub[metric], color="steelblue", alpha=0.8)
        ax.set_title(target, fontsize=9)
        ax.set_xticklabels(sub["model"], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Ranking {metric} por Target", fontsize=13)
    plt.tight_layout()
    _save(fig, out_dir / f"ranking_{metric.lower()}.png")
