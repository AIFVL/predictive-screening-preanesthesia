# src/reports/shap_plots.py
"""
Generación de plots SHAP: beeswarm global y waterfall por caso.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("reports.shap_plots")

# Número máximo de features en el beeswarm
_BEESWARM_TOP_N = 20
# Número máximo de waterfall plots (casos FN)
_WATERFALL_MAX_CASES = 10


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    model_key: str,
    target_name: str,
    output_dir: Path,
    top_n: int = _BEESWARM_TOP_N,
) -> None:
    """
    Genera el beeswarm plot SHAP global y lo guarda como PNG.

    El beeswarm muestra, para cada feature, un punto por paciente:
    - Posición horizontal: cuánto empujó la predicción hacia positivo (+) o negativo (−).
    - Color: valor de la feature (rojo=alto, azul=bajo).
    """
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"shap_beeswarm_{model_key}.png"

    # Seleccionar top_n features por importancia media absoluta
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    feature_names = list(X_explain.columns)

    sv_top = shap_values[:, top_idx]
    X_top = X_explain.iloc[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    shap.summary_plot(
        sv_top,
        X_top,
        feature_names=[feature_names[i] for i in top_idx],
        plot_type="dot",
        show=False,
        max_display=top_n,
    )
    plt.title(f"SHAP Beeswarm — {model_key} / {target_name}", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Beeswarm guardado: {out_path}")


def plot_shap_waterfall_fn(
    shap_values: np.ndarray,
    expected_value: float,
    X_explain: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_key: str,
    target_name: str,
    output_dir: Path,
    max_cases: int = _WATERFALL_MAX_CASES,
) -> None:
    """
    Genera un waterfall plot por cada falso negativo (FN), ordenados de mayor
    a menor probabilidad predicha (los más "difíciles" de detectar primero).

    Cada plot muestra la contribución de cada feature a la predicción de ese
    paciente, partiendo del valor base (expected_value) hasta llegar a la
    probabilidad final del modelo.
    """
    import shap

    fn_dir = output_dir / "fn_waterfall"
    fn_dir.mkdir(parents=True, exist_ok=True)

    # Identificar FN: positivos reales que el modelo no detectó
    y_true_arr = np.asarray(y_true)
    fn_mask = (y_true_arr == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]

    if len(fn_indices) == 0:
        logger.info("  No hay FN en este conjunto — waterfall omitido.")
        return

    # Ordenar FN por probabilidad predicha descendente (más cercanos al umbral primero)
    fn_proba = y_proba[fn_indices]
    order = np.argsort(fn_proba)[::-1]
    fn_indices_sorted = fn_indices[order][:max_cases]

    feature_names = list(X_explain.columns)

    for rank, local_idx in enumerate(fn_indices_sorted):
        sv_case = shap_values[local_idx]
        x_case = X_explain.iloc[local_idx]
        proba_case = y_proba[local_idx]

        # Construir Explanation object de shap para waterfall
        explanation = shap.Explanation(
            values=sv_case,
            base_values=expected_value,
            data=x_case.values,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.25)))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(
            f"FN #{rank + 1} — {model_key} / {target_name}\n"
            f"P(positivo) = {proba_case:.3f} | idx={local_idx}",
            fontsize=10,
            pad=10,
        )
        plt.tight_layout()
        out_path = fn_dir / f"waterfall_fn_{rank + 1:02d}_{model_key}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")

    logger.info(
        f"  {min(len(fn_indices_sorted), max_cases)} waterfall FN guardados en {fn_dir}"
    )


def save_shap_values(
    shap_values: np.ndarray,
    expected_value: float,
    X_explain: pd.DataFrame,
    output_dir: Path,
    model_key: str,
) -> None:
    """
    Persiste los SHAP values crudos para uso posterior (análisis, reportes).

    Guarda:
    - shap_values_{model_key}.npy   — array (n_samples, n_features)
    - shap_expected_{model_key}.txt — escalar float
    - shap_features_{model_key}.txt — nombres de features (uno por línea)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"shap_values_{model_key}.npy", shap_values)
    (output_dir / f"shap_expected_{model_key}.txt").write_text(str(expected_value))
    (output_dir / f"shap_features_{model_key}.txt").write_text(
        "\n".join(X_explain.columns.tolist())
    )
    logger.info(f"  SHAP values crudos guardados en {output_dir}")
