# src/analysis/flag_predictability.py
"""
Enfoque C: ¿Qué flags posoperatorios son predecibles desde variables preoperatorias?

Para cada flag con prevalencia >= min_prevalence, entrena un RandomForestClassifier
sobre las features preoperatorias del merged y calcula ROC AUC en CV estratificada.
Produce un ranking de flags por predictibilidad para orientar la redefinición del target.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.logger import get_logger

logger = get_logger("analysis.flag_predictability")

# Flags excluidos: son proceso/técnica, no complicaciones clínicas
_FLAGS_EXCLUIR = {
    "flag_fisiologicas",
    "flag_ventilacion",
    "flag_tecnica",
    "flag_induccion",
    "flag_tiempos",
    "flag_hemoderivados",
    "flag_complicacion",
    "flag_cancelacion",
    "flag_monitoreo_invasivo",
    "flag_tecnica_combinada",
    "flag_hoja_laringoscopio_recta",
    "flag_control_manual",
    "flag_ventilacion_asistida",
    "flag_modos_avanzados",
    "flag_parametros_ventilatorios",
    "flag_frecuencia_resp_anormal",
    "flag_no_despierto",
    "flag_desenlace",
    "flag_estancia",
    "flag_reservas",
}


def run_flag_predictability(
    merged_path: str | Path,
    output_dir: str | Path,
    min_prevalence: float = 0.01,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula ROC AUC de cada flag posoperatorio usando features preoperatorias.

    Parámetros:
        merged_path:     path a merged.parquet (preop + flags posop + target)
        output_dir:      directorio de salida para CSV y PNG
        min_prevalence:  prevalencia mínima del flag para incluirlo
        n_folds:         folds para cross-validation estratificada
        random_state:    semilla

    Retorna DataFrame con columnas:
        flag, prevalence, n_positives, roc_auc_mean, roc_auc_std, interpretacion
    """
    merged_path = Path(merged_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(merged_path)
    if "Edad" in df.columns:
        df = df[df["Edad"] >= 18].reset_index(drop=True)

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    exclude = set(flag_cols) | {"target", "Documento PMD", "Documento_PMD", "n_flags_relevant"}
    pre_cols = [c for c in df.columns if c not in exclude]

    X = df[pre_cols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() >= 0.4]
    X = X.fillna(X.median())

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=20,
        random_state=random_state, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    rows = []
    flags_to_analyze = [f for f in flag_cols if f not in _FLAGS_EXCLUIR]

    for flag in sorted(flags_to_analyze):
        if flag not in df.columns:
            continue
        y = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
        prevalence = float(y.mean())
        n_positives = int(y.sum())

        if prevalence < min_prevalence or y.nunique() < 2:
            continue

        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        roc_mean = float(scores.mean())
        roc_std = float(scores.std())

        if roc_mean >= 0.75:
            interpretacion = "buena_senal"
        elif roc_mean >= 0.65:
            interpretacion = "senal_moderada"
        elif roc_mean >= 0.55:
            interpretacion = "senal_debil"
        else:
            interpretacion = "sin_senal"

        logger.info(
            f"  {flag:<45} prev={prevalence:.3f}  "
            f"ROC={roc_mean:.3f}±{roc_std:.3f}  [{interpretacion}]"
        )
        rows.append({
            "flag": flag,
            "prevalence": round(prevalence, 4),
            "n_positives": n_positives,
            "roc_auc_mean": round(roc_mean, 4),
            "roc_auc_std": round(roc_std, 4),
            "interpretacion": interpretacion,
        })

    df_result = (
        pd.DataFrame(rows)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / "flag_predictability.csv"
    df_result.to_csv(csv_path, index=False)
    logger.info(f"Exportado: {csv_path}")

    # ── PNG ───────────────────────────────────────────────────────────────────
    colors = {
        "buena_senal": "#2ecc71",
        "senal_moderada": "#f39c12",
        "senal_debil": "#e74c3c",
        "sin_senal": "#bdc3c7",
    }
    bar_colors = [colors[i] for i in df_result["interpretacion"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_result) * 0.4)))
    ax.barh(
        df_result["flag"], df_result["roc_auc_mean"],
        xerr=df_result["roc_auc_std"],
        color=bar_colors, capsize=3, alpha=0.85,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Azar (0.5)")
    ax.axvline(0.65, color="gray", linestyle=":", linewidth=1, label="Señal moderada (0.65)")
    ax.axvline(0.75, color="#2ecc71", linestyle=":", linewidth=1, label="Buena señal (0.75)")

    for i, (_, row) in enumerate(df_result.iterrows()):
        ax.text(0.505, i, f"prev={row['prevalence']:.1%}", va="center", fontsize=7, color="#555")

    ax.set_xlabel("ROC AUC (CV 5-fold)")
    ax.set_title(
        "Predictibilidad de cada flag posoperatorio\ndesde variables preoperatorias",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.45, 1.0)
    plt.tight_layout()
    png_path = output_dir / "flag_predictability.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    logger.info(f"Exportado: {png_path}")

    for interp in ["buena_senal", "senal_moderada", "senal_debil", "sin_senal"]:
        n = (df_result["interpretacion"] == interp).sum()
        logger.info(f"  {interp}: {n} flags")

    return df_result
