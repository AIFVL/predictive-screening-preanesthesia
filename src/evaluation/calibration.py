# src/evaluation/calibration.py
"""
Calibración probabilística de modelos: reliability diagram + ECE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.utils.logger import get_logger

logger = get_logger("evaluation.calibration")


def compute_calibration_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """
    Calcula la curva de calibración y métricas derivadas.

    Parámetros
    ----------
    y_true    : etiquetas reales (0/1).
    y_proba   : probabilidades predichas para la clase positiva.
    n_bins    : número de bins (10 = resolución estándar clínica).
    strategy  : 'uniform' (bins de igual ancho) o 'quantile' (bins de igual frecuencia).

    Retorna
    -------
    dict con claves:
        fraction_of_positives : array — proporción real de positivos por bin.
        mean_predicted        : array — probabilidad media predicha por bin.
        bin_counts            : array — número de muestras por bin.
        ece                   : float — Expected Calibration Error.
        mce                   : float — Maximum Calibration Error (bin más desviado).
        brier                 : float — Brier score (MSE de probabilidades).
        overconfident_bins    : int   — bins donde el modelo sobreestima.
        underconfident_bins   : int   — bins donde el modelo subestima.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy=strategy
    )

    # Contar muestras por bin para ponderar el ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.zeros(len(frac_pos), dtype=int)
    for i in range(len(bin_edges) - 1):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if i < len(bin_counts):
            bin_counts[i] = mask.sum()
    # Último bin incluye el 1.0
    if len(bin_counts) > 0:
        bin_counts[-1] += (y_proba == 1.0).sum()

    n_total = len(y_true)
    # ECE: promedio ponderado del error absoluto por bin
    errors = np.abs(frac_pos - mean_pred)
    weights = bin_counts / n_total if n_total > 0 else np.ones(len(bin_counts)) / len(bin_counts)
    ece = float(np.sum(errors * weights))
    mce = float(errors.max()) if len(errors) > 0 else 0.0

    brier = float(np.mean((y_proba - y_true) ** 2))

    overconfident  = int((mean_pred > frac_pos).sum())
    underconfident = int((mean_pred < frac_pos).sum())

    return {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted":        mean_pred.tolist(),
        "bin_counts":            bin_counts.tolist(),
        "ece":                   round(ece, 6),
        "mce":                   round(mce, 6),
        "brier":                 round(brier, 6),
        "overconfident_bins":    overconfident,
        "underconfident_bins":   underconfident,
        "n_bins":                n_bins,
        "strategy":              strategy,
        "n_samples":             n_total,
    }


def calibration_summary(results: dict[str, dict]) -> pd.DataFrame:
    """
    Recibe {model_key: calibration_dict} y devuelve un DataFrame comparativo.
    Útil para el plot de comparación multi-modelo.
    """
    rows = []
    for model, data in results.items():
        rows.append({
            "model":               model,
            "ECE":                 data["ece"],
            "MCE":                 data["mce"],
            "Brier":               data["brier"],
            "overconfident_bins":  data["overconfident_bins"],
            "underconfident_bins": data["underconfident_bins"],
        })
    return pd.DataFrame(rows).sort_values("ECE").reset_index(drop=True)
