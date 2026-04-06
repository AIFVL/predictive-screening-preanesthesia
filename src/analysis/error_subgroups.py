# src/analysis/error_subgroups.py
"""
Enfoque B: ¿Dónde falla el modelo y dónde funciona bien?

Carga el modelo random_forest de target_d_v2_hosp, obtiene predicciones sobre
el test set, y calcula métricas (ROC AUC, FN rate, FP rate) por subgrupos
de variables preoperatorias.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.logger import get_logger

logger = get_logger("analysis.error_subgroups")


def _metrics_for_subgroup(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    if len(y_true) < 10 or y_true.nunique() < 2:
        return {"roc_auc": float("nan"), "fn_rate": float("nan"), "fp_rate": float("nan")}

    y_pred = (y_proba >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    fn_rate = float((y_pred[positives] == 0).mean()) if positives.sum() > 0 else float("nan")
    fp_rate = float((y_pred[negatives] == 1).mean()) if negatives.sum() > 0 else float("nan")
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    return {"roc_auc": auc, "fn_rate": fn_rate, "fp_rate": fp_rate}


def run_error_subgroups(
    merged_path: str | Path,
    splits_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.17,
) -> pd.DataFrame:
    """
    Analiza errores del modelo random_forest por subgrupos preoperatorios.

    Parámetros:
        merged_path:  merged.parquet de target_d_v2_hosp
        splits_dir:   directorio con X_test.parquet y y_test.parquet
        model_path:   .joblib del modelo random_forest
        output_dir:   directorio de salida
        threshold:    0.17 (valor guardado en random_forest_metrics.json)

    Retorna DataFrame con columnas:
        variable, subgrupo, n, n_positivos, prevalence, roc_auc, fn_rate, fp_rate
    """
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_parquet(splits_dir / "X_test.parquet")
    y_test = pd.read_parquet(splits_dir / "y_test.parquet").squeeze()
    model = joblib.load(model_path)

    X_test_num = X_test.apply(pd.to_numeric, errors="coerce").fillna(-1)
    y_proba = model.predict_proba(X_test_num)[:, 1]

    df_merged = pd.read_parquet(merged_path)
    if "Edad" in df_merged.columns:
        df_merged = df_merged[df_merged["Edad"] >= 18]

    df_ctx = df_merged.loc[X_test.index].copy()
    df_ctx["y_true"] = y_test.values
    df_ctx["y_proba"] = y_proba

    rows = []

    def _add_subgroup(variable, subgrupo, mask):
        if mask.sum() < 20:
            return
        sub = df_ctx[mask]
        m = _metrics_for_subgroup(sub["y_true"], sub["y_proba"].values, threshold)
        rows.append({
            "variable": variable,
            "subgrupo": subgrupo,
            "n": int(mask.sum()),
            "n_positivos": int(sub["y_true"].sum()),
            "prevalence": round(float(sub["y_true"].mean()), 4),
            "roc_auc": round(m["roc_auc"], 4),
            "fn_rate": round(m["fn_rate"], 4),
            "fp_rate": round(m["fp_rate"], 4),
        })

    # ── Tipo de anestesia propuesta ───────────────────────────────────────────
    for col in [c for c in df_ctx.columns if c.startswith("Tipo de anestesia propuesta_")]:
        label = col.replace("Tipo de anestesia propuesta_", "")
        _add_subgroup("tipo_anestesia", label, df_ctx[col] == 1)

    # ── Score de severidad del procedimiento (quartiles) ─────────────────────
    if "score_proc_high_severity" in df_ctx.columns:
        q = pd.qcut(
            df_ctx["score_proc_high_severity"], q=4,
            labels=["Q1_bajo", "Q2", "Q3", "Q4_alto"], duplicates="drop",
        )
        for qval in ["Q1_bajo", "Q2", "Q3", "Q4_alto"]:
            _add_subgroup("score_proc_severidad_quartil", qval, q == qval)

    # ── Grupos de edad ────────────────────────────────────────────────────────
    if "Edad" in df_ctx.columns:
        bins = [18, 40, 60, 75, 120]
        labels_edad = ["18-40", "41-60", "61-75", "76+"]
        edad_group = pd.cut(df_ctx["Edad"], bins=bins, labels=labels_edad, include_lowest=True)
        for eg in labels_edad:
            _add_subgroup("grupo_edad", eg, edad_group == eg)

    # ── Capítulo CIE-10 del diagnóstico ──────────────────────────────────────
    for col in [c for c in df_ctx.columns if c.startswith("Dx Preoperatorio Code_")]:
        label = col.replace("Dx Preoperatorio Code_", "")
        _add_subgroup("dx_cie10_capitulo", label, df_ctx[col] == 1)

    df_result = pd.DataFrame(rows).sort_values(["variable", "roc_auc"], ascending=[True, False])

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / "error_analysis.csv"
    df_result.to_csv(csv_path, index=False)
    logger.info(f"Exportado: {csv_path} ({len(df_result)} subgrupos)")

    # ── PNG: grilla 2x2 con las 4 variables de segmentación ──────────────────
    variables = ["tipo_anestesia", "score_proc_severidad_quartil", "grupo_edad", "dx_cie10_capitulo"]
    global_auc = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, variable in zip(axes, variables):
        sub = df_result[df_result["variable"] == variable].sort_values("roc_auc", ascending=True)
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.barh(sub["subgrupo"], sub["roc_auc"], color="#3498db", alpha=0.8)
        ax.axvline(global_auc, color="red", linestyle="--", linewidth=1.5,
                   label=f"Global ROC={global_auc:.3f}")
        ax.set_xlabel("ROC AUC")
        ax.set_title(f"ROC AUC por {variable}", fontweight="bold")
        ax.set_xlim(0.4, 1.0)
        ax.legend(fontsize=8)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(0.41, i, f"n={row['n']}", va="center", fontsize=7)

    fig.suptitle(
        "Análisis de error por subgrupos preoperatorios\n"
        "(modelo random_forest, target_d_v2_hosp, threshold=0.17)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    png_path = output_dir / "error_subgroups.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    logger.info(f"Exportado: {png_path}")

    logger.info(f"ROC AUC global en test: {global_auc:.4f}")
    logger.info("Top 3 mejores subgrupos:")
    logger.info(df_result.nlargest(3, "roc_auc")[["variable","subgrupo","roc_auc","n"]].to_string())
    logger.info("Top 3 peores subgrupos:")
    logger.info(df_result.nsmallest(3, "roc_auc")[["variable","subgrupo","roc_auc","n"]].to_string())

    return df_result
