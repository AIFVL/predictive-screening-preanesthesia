# src/reports/shap_plots.py
"""
Generación de plots SHAP: beeswarm global, waterfall por caso y comparación FN vs TP.
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

    # Normalizar a 2D — CalibratedClassifierCV puede devolver (n, f, n_classes)
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    # Seleccionar top_n features por importancia media absoluta
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.ravel(np.argsort(mean_abs)[::-1][:top_n])  # garantizar 1D
    feature_names = list(X_explain.columns)

    sv_top = sv[:, top_idx]
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

    # Normalizar a 2D
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

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
    # Guardar siempre en 2D — normalizar si viene 3D de CalibratedClassifierCV
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    np.save(output_dir / f"shap_values_{model_key}.npy", sv)
    (output_dir / f"shap_expected_{model_key}.txt").write_text(str(expected_value))
    (output_dir / f"shap_features_{model_key}.txt").write_text(
        "\n".join(X_explain.columns.tolist()), encoding="utf-8"
    )
    logger.info(f"  SHAP values crudos guardados en {output_dir}")


def plot_shap_group_comparison(
    profiles_df,
    model_key: str,
    target_name: str,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Barras horizontales comparando SHAP medio de FN vs TP para las top_n features
    con mayor diferencia absoluta entre ambos grupos.

    Convención de colores:
        Verde  (#2ca02c) = TP
        Naranja (#ff7f0e) = FN
    Una barra más larga hacia la derecha = esa feature empuja más hacia positivo en ese grupo.
    Una barra hacia la izquierda = esa feature reduce la probabilidad en ese grupo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = profiles_df.head(top_n).copy()
    features = df["Feature"].tolist()
    shap_tp  = df["SHAP_TP"].values
    shap_fn  = df["SHAP_FN"].values
    delta    = df["Delta_TP_FN"].values

    n_tp = int(df["N_TP"].iloc[0]) if "N_TP" in df.columns else "?"
    n_fn = int(df["N_FN"].iloc[0]) if "N_FN" in df.columns else "?"

    y = np.arange(len(features))
    bar_h = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, top_n * 0.55)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Panel izquierdo: barras TP vs FN ──────────────────────────────────
    ax = axes[0]
    bars_tp = ax.barh(y + bar_h / 2, shap_tp, bar_h,
                      color="#2ca02c", alpha=0.8, label=f"TP (n={n_tp})")
    bars_fn = ax.barh(y - bar_h / 2, shap_fn, bar_h,
                      color="#ff7f0e", alpha=0.8, label=f"FN (n={n_fn})")

    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f[:45] + ("…" if len(f) > 45 else "") for f in features],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("SHAP medio (impacto en probabilidad)", fontsize=11)
    ax.set_title(
        f"Perfil SHAP: TP vs FN\n{model_key} / {target_name}",
        fontsize=12, pad=10,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    # Sombreado para destacar dirección opuesta entre grupos
    for i, (tp, fn) in enumerate(zip(shap_tp, shap_fn)):
        if np.sign(tp) != np.sign(fn):
            ax.axhspan(i - 0.5, i + 0.5, alpha=0.06, color="purple")

    # ── Panel derecho: delta TP - FN ──────────────────────────────────────
    ax2 = axes[1]
    colors_delta = ["#2ca02c" if d > 0 else "#ff7f0e" for d in delta]
    ax2.barh(y, delta, 0.6, color=colors_delta, alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Delta (TP − FN)", fontsize=11)
    ax2.set_title("Diferencia\nTP − FN", fontsize=11, pad=10)
    ax2.grid(True, axis="x", alpha=0.3)

    # Anotaciones de valor delta
    for i, d in enumerate(delta):
        ax2.text(d + (0.001 if d >= 0 else -0.001), i,
                 f"{d:+.3f}", va="center",
                 ha="left" if d >= 0 else "right",
                 fontsize=8, color="black")

    # Leyenda de sombreado
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="purple", alpha=0.15, label="Dirección opuesta TP/FN"),
        Patch(facecolor="#2ca02c", alpha=0.7, label="Delta positivo = TP tiene más señal"),
        Patch(facecolor="#ff7f0e", alpha=0.7, label="Delta negativo = FN tiene más señal"),
    ]
    axes[0].legend(handles=legend_elements + list(axes[0].get_legend_handles_labels()[0]),
                   fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = output_dir / f"shap_group_comparison_{model_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"  Comparación SHAP FN vs TP guardada: {out_path}")
