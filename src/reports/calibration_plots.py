# src/reports/calibration_plots.py
"""
Reliability diagrams y comparación de calibración entre modelos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("reports.calibration_plots")

# Paleta consistente con el resto del proyecto
_MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_calibration_curve(
    calibration_data: dict,
    model_key: str,
    target_name: str,
    output_dir: Path,
) -> None:
    """
    Reliability diagram para un único modelo.

    Muestra:
    - Línea diagonal: calibración perfecta.
    - Puntos: proporción real de positivos por bin (tamaño ∝ n muestras del bin).
    - Histograma inferior: distribución de las probabilidades predichas.
    - ECE y Brier en el título.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frac_pos  = np.array(calibration_data["fraction_of_positives"])
    mean_pred = np.array(calibration_data["mean_predicted"])
    counts    = np.array(calibration_data["bin_counts"])
    ece       = calibration_data["ece"]
    brier     = calibration_data["brier"]

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Reliability diagram ────────────────────────────────────────────────
    ax_main.plot([0, 1], [0, 1], "k--", lw=1.2, label="Calibración perfecta", alpha=0.6)

    # Región de sobreconfianza / subconfianza
    ax_main.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="red",
                         label="Zona sobreconfianza")
    ax_main.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="blue",
                         label="Zona subconfianza")

    # Puntos calibración (tamaño ∝ n muestras)
    max_count = counts.max() if counts.max() > 0 else 1
    sizes = 40 + 200 * (counts / max_count)
    sc = ax_main.scatter(
        mean_pred, frac_pos,
        s=sizes, c=mean_pred, cmap="RdYlGn_r",
        edgecolors="black", linewidths=0.6, zorder=5,
    )
    ax_main.plot(mean_pred, frac_pos, "-o", color="#1f77b4",
                 markersize=4, lw=1.5, alpha=0.7)

    ax_main.set_ylabel("Fracción real de positivos", fontsize=12)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_title(
        f"Reliability Diagram — {model_key} / {target_name}\n"
        f"ECE = {ece:.4f}   |   Brier = {brier:.4f}",
        fontsize=12, pad=10,
    )
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.3)

    # Anotaciones de dirección
    over  = calibration_data["overconfident_bins"]
    under = calibration_data["underconfident_bins"]
    ax_main.text(0.97, 0.03,
                 f"Bins sobreconfiados: {over}\nBins subconfiados: {under}",
                 transform=ax_main.transAxes, ha="right", va="bottom",
                 fontsize=9, color="gray",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Histograma de probabilidades ───────────────────────────────────────
    ax_hist.bar(
        mean_pred, counts, width=0.08,
        color="#1f77b4", alpha=0.6, edgecolor="white",
    )
    ax_hist.set_xlabel("Probabilidad predicha", fontsize=12)
    ax_hist.set_ylabel("N pacientes", fontsize=10)
    ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax_hist.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"calibration_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Reliability diagram guardado: {out_path}")


def plot_calibration_comparison(
    all_calibrations: dict[str, dict],
    target_name: str,
    output_dir: Path,
) -> None:
    """
    Reliability diagram con todos los modelos superpuestos + tabla ECE/Brier.

    Permite identificar de un vistazo qué modelo está mejor calibrado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_diag, ax_table) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    ax_diag.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfecta", alpha=0.7, zorder=10)

    rows_table = []
    for i, (model, data) in enumerate(sorted(all_calibrations.items())):
        frac_pos  = np.array(data["fraction_of_positives"])
        mean_pred = np.array(data["mean_predicted"])
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        ax_diag.plot(
            mean_pred, frac_pos,
            "o-", color=color, lw=1.8, markersize=5,
            label=f"{model}  (ECE={data['ece']:.4f})",
            alpha=0.85,
        )
        rows_table.append({
            "Modelo":  model,
            "ECE":     f"{data['ece']:.4f}",
            "Brier":   f"{data['brier']:.4f}",
            "Sobre":   data["overconfident_bins"],
            "Bajo":    data["underconfident_bins"],
        })

    ax_diag.set_xlabel("Probabilidad predicha", fontsize=12)
    ax_diag.set_ylabel("Fracción real de positivos", fontsize=12)
    ax_diag.set_title(f"Calibración comparativa — {target_name}", fontsize=13)
    ax_diag.set_xlim(-0.02, 1.02)
    ax_diag.set_ylim(-0.05, 1.05)
    ax_diag.legend(fontsize=8, loc="upper left")
    ax_diag.grid(True, alpha=0.3)

    # Tabla resumen
    df_table = pd.DataFrame(rows_table).sort_values("ECE")
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    # Header oscuro
    for j in range(len(df_table.columns)):
        tbl[0, j].set_facecolor("#1A3C34")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Mejor ECE en verde
    best_row = df_table.index[0]  # ya ordenado
    for j in range(len(df_table.columns)):
        tbl[1, j].set_facecolor("#E8F5D0")

    ax_table.set_title("Resumen calibración\n(ordenado por ECE)", fontsize=11)

    plt.tight_layout()
    out_path = output_dir / "calibration_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Comparación de calibración guardada: {out_path}")
