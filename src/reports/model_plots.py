# src/reports/model_plots.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

from src.utils.logger import get_logger

logger = get_logger("reports.model_plots")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot: {path.name}")


def plot_roc_pr(
    y_true,
    y_proba,
    model_name: str,
    target_name: str,
    out_dir: Path | str,
) -> None:
    """ROC + Precision-Recall curves para un modelo."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR (Recall)")
    axes[0].set_title(f"ROC — {model_name} [{target_name}]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall_vals, precision_vals, linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR Curve — {model_name} [{target_name}]")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir / f"{model_name}_roc_pr.png")


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    target_name: str,
    out_dir: Path | str,
) -> None:
    """Matriz de confusión."""
    import seaborn as sns
    out_dir = Path(out_dir)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Real 0", "Real 1"])
    ax.set_title(f"Confusion Matrix — {model_name} [{target_name}]")
    plt.tight_layout()
    _save(fig, out_dir / f"{model_name}_confusion.png")


def plot_threshold_curve(
    thresholds: list[float],
    scores: list[float],
    optimal_threshold: float,
    model_name: str,
    target_name: str,
    out_dir: Path | str,
    metric_label: str = "F2",
) -> None:
    """Curva de métrica vs threshold con línea en óptimo."""
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, scores, linewidth=2, color="steelblue")
    ax.axvline(optimal_threshold, color="red", linestyle="--",
               label=f"Óptimo: {optimal_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default: 0.50")
    ax.set_xlabel("Umbral")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Threshold — {model_name} [{target_name}]")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / f"{model_name}_threshold.png")
