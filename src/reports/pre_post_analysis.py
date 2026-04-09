# src/reports/pre_post_analysis.py
"""
Análisis de señal pre→post: Información Mutua + Correlación de Pearson por target/flag.
Migrado fielmente desde utils/multi_version/pre_post_analysis.py.
Adaptación: I/O en Parquet (no Excel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils.io import read_parquet
from src.utils.logger import get_logger

logger = get_logger("reports.pre_post_analysis")

_EXCLUDED_EXACT = {"target", "Documento PMD", "Documento_PMD", "n_flags_relevant"}
_MIN_NUMERIC_RATIO = 0.4


def _prepare_numeric_pre_features(df: pd.DataFrame, min_non_null_ratio: float = 0.7) -> pd.DataFrame:
    """
    Selecciona columnas numéricas válidas del merged DataFrame para ser usadas como
    features preoperatorias. Excluye 'Documento PMD' y columnas con demasiados nulos.
    Imputa nulos restantes con la mediana.
    """
    numeric_candidates = []
    for col in df.columns:
        if col == "Documento PMD":
            continue
        series_num = pd.to_numeric(df[col], errors="coerce")
        non_null_ratio = float(series_num.notna().mean())
        if non_null_ratio >= min_non_null_ratio:
            numeric_candidates.append(series_num.rename(col))

    if not numeric_candidates:
        return pd.DataFrame(index=df.index)

    df_num = pd.concat(numeric_candidates, axis=1)
    for col in df_num.columns:
        median_value = df_num[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        df_num[col] = df_num[col].fillna(median_value)

    return df_num


def _compute_linkage_for_post_variable(
    X_pre: pd.DataFrame,
    y_post: pd.Series,
    version: str,
    post_variable: str,
    top_n: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Calcula MI + Pearson para una variable posoperatoria contra las features preoperatorias.
    Combined_Score = 0.7 * MI + 0.3 * |Pearson|.
    """
    if y_post.nunique(dropna=True) < 2 or X_pre.empty:
        return pd.DataFrame(columns=[
            "Version", "Post_Variable", "Feature",
            "Mutual_Information", "Abs_Pearson", "Combined_Score",
        ])

    mi = mutual_info_classif(X_pre, y_post, random_state=random_state)
    corr = X_pre.corrwith(y_post).abs().fillna(0.0)

    result = pd.DataFrame(
        {
            "Version": version,
            "Post_Variable": post_variable,
            "Feature": X_pre.columns,
            "Mutual_Information": mi,
            "Abs_Pearson": corr.values,
        }
    )
    result["Combined_Score"] = 0.7 * result["Mutual_Information"] + 0.3 * result["Abs_Pearson"]

    result = result.sort_values(
        ["Combined_Score", "Mutual_Information"], ascending=False
    ).reset_index(drop=True)

    return result.head(top_n).copy() if top_n is not None else result.copy()


def run_pre_post_linkage_analysis(
    proc_dir: Path,
    output_dir: Path,
    active_targets: list[str],
    post_variables: list[str] | None = None,
    top_n: int | None = None,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Analiza la relación entre variables preoperatorias y variables posoperatorias
    (flags individuales + target) usando Información Mutua y Correlación de Pearson.

    Lee merged.parquet de cada target desde proc_dir/<target>/merged.parquet.
    Escribe resultados en output_dir/pre_post_signal/.

    Parámetros:
        proc_dir:       directorio con subdirectorios por target, cada uno con merged.parquet
        output_dir:     directorio donde se exportan los CSVs
        active_targets: lista de nombres de targets a procesar
        post_variables: columnas POS a analizar (None = auto-detecta flag_* + target)
        top_n:          features PRE a mostrar por variable POS (None = todas)
        random_state:   semilla para Información Mutua

    Retorna dict con:
        "per_flag_linkage": ranking de features PRE por cada flag POS y versión
        "flag_summary":     resumen de señal por flag (max MI, top-3 features, prevalencia)
    """
    proc_dir = Path(proc_dir)
    signal_dir = Path(output_dir) / "pre_post_signal"
    signal_dir.mkdir(parents=True, exist_ok=True)

    all_linkage_rows: list[pd.DataFrame] = []

    for target in active_targets:
        merged_path = proc_dir / target / "merged.parquet"
        if not merged_path.exists():
            logger.warning(f"[{target}] merged.parquet no encontrado — omitido.")
            continue

        df_merged = read_parquet(merged_path)

        if "Edad" in df_merged.columns:
            df_merged = df_merged[df_merged["Edad"] >= 18].reset_index(drop=True)

        # Features preoperatorias: columnas numéricas sin flags ni target
        known_pos = {c for c in df_merged.columns if c.startswith("flag_") or c == "target"}
        pre_cols = []
        for col in df_merged.columns:
            if col in _EXCLUDED_EXACT or col in known_pos:
                continue
            series_num = pd.to_numeric(df_merged[col], errors="coerce")
            if series_num.notna().mean() < _MIN_NUMERIC_RATIO:
                continue
            if series_num.nunique(dropna=True) <= 1:
                continue
            pre_cols.append(col)

        if not pre_cols:
            logger.warning(f"[{target}] Sin features preoperatorias válidas — omitido.")
            continue

        X_pre = df_merged[pre_cols].apply(pd.to_numeric, errors="coerce").fillna(-1)

        # Variables posoperatorias
        if post_variables is not None:
            pos_cols = [c for c in post_variables if c in df_merged.columns]
        else:
            pos_cols = sorted(
                [c for c in df_merged.columns if c.startswith("flag_")]
                + (["target"] if "target" in df_merged.columns else [])
            )

        if not pos_cols:
            logger.warning(f"[{target}] Sin columnas posoperatorias — omitido.")
            continue

        logger.info(f"[{target}] Features PRE: {len(pre_cols)} | Variables POS: {len(pos_cols)}")

        for pos_var in pos_cols:
            y_post = pd.to_numeric(df_merged[pos_var], errors="coerce").dropna().astype(int)
            if y_post.nunique() < 2:
                continue
            X_aligned = X_pre.loc[y_post.index]
            prevalence = float(y_post.mean())

            linkage = _compute_linkage_for_post_variable(
                X_aligned, y_post, target, pos_var, top_n=top_n, random_state=random_state
            )
            if not linkage.empty:
                linkage["Prevalencia_Post"] = round(prevalence, 4)
                all_linkage_rows.append(linkage)

            informative = int((linkage["Mutual_Information"] > 0.01).sum()) if not linkage.empty else 0
            max_mi = linkage["Mutual_Information"].max() if not linkage.empty else 0.0
            logger.info(
                f"  {pos_var:<45} prevalencia={prevalence:.1%}  "
                f"max_MI={max_mi:.4f}  n_informativas={informative}"
            )

    if not all_linkage_rows:
        raise ValueError("No se pudo calcular ningún linkage PRE→POS.")

    df_linkage = pd.concat(all_linkage_rows, ignore_index=True)

    summary_rows = []
    for (version, pos_var), group in df_linkage.groupby(["Version", "Post_Variable"]):
        top3 = group.nlargest(3, "Mutual_Information")["Feature"].tolist()
        summary_rows.append({
            "Version": version,
            "Post_Variable": pos_var,
            "Prevalencia": group["Prevalencia_Post"].iloc[0],
            "Max_MI": round(group["Mutual_Information"].max(), 5),
            "Max_Pearson": round(group["Abs_Pearson"].max(), 5),
            "N_Features_Informativas": int((group["Mutual_Information"] > 0.01).sum()),
            "Top3_Features_PRE": " | ".join(top3),
        })

    df_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["Version", "Max_MI"], ascending=[True, False])
        .reset_index(drop=True)
    )

    linkage_path = signal_dir / "pre_post_linkage_per_flag.csv"
    summary_path = signal_dir / "pre_post_linkage_summary.csv"
    df_linkage.to_csv(linkage_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Exportado: {linkage_path}")
    logger.info(f"Exportado: {summary_path}")

    return {"per_flag_linkage": df_linkage, "flag_summary": df_summary}
