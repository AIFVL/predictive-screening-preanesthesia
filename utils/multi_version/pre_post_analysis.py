from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

_EXCLUDED_EXACT = {"target", "Documento PMD", "Documento_PMD", "n_flags_relevant"}
_MIN_NUMERIC_RATIO = 0.4


def _resolve_active_pos_files(data_dir: Path, active_versions: list[str] | None) -> dict[str, Path]:
    pos_files = sorted(data_dir.glob("OPERA_POS_*.xlsx"))
    resolved = {}

    for path in pos_files:
        version = path.stem.replace("OPERA_POS_", "")
        if active_versions is None or version in set(active_versions):
            resolved[version] = path

    return resolved


def _prepare_numeric_pre_features(df_pre: pd.DataFrame, min_non_null_ratio: float = 0.7) -> pd.DataFrame:
    numeric_candidates = []
    for col in df_pre.columns:
        if col == "Documento PMD":
            continue
        series_num = pd.to_numeric(df_pre[col], errors="coerce")
        non_null_ratio = float(series_num.notna().mean())
        if non_null_ratio >= min_non_null_ratio:
            numeric_candidates.append(series_num.rename(col))

    if not numeric_candidates:
        return pd.DataFrame(index=df_pre.index)

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
    top_n: int,
    random_state: int,
) -> pd.DataFrame:
    if y_post.nunique(dropna=True) < 2 or X_pre.empty:
        return pd.DataFrame(columns=[
            "Version",
            "Post_Variable",
            "Feature",
            "Mutual_Information",
            "Abs_Pearson",
            "Combined_Score",
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

    result = result.sort_values(["Combined_Score", "Mutual_Information"], ascending=False).reset_index(drop=True)
    return result.head(top_n).copy() if top_n is not None else result.copy()


def run_pre_post_linkage_analysis(
    merged_dir: Path,
    output_dir: Path,
    pos_dir: Path | None = None,
    post_variables: list[str] | None = None,
    top_n: int | None = None,
    random_state: int = 42,
    active_versions: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Analiza la relación entre variables preoperatorias y variables posoperatorias
    (flags individuales + target) usando Información Mutua y Correlación de Pearson.

    Permite evaluar ANTES de modelar qué señal existe en el preoperatorio para predecir
    cada tipo de complicación posoperatoria por separado.

    Parámetros:
        merged_dir:     directorio con archivos OPERA_COMPLETO_*.xlsx (features PRE + target)
        output_dir:     directorio donde se exportan los CSVs de resultados
        pos_dir:        directorio con archivos OPERA_POS_*.xlsx (flags individuales POS).
                        Si se provee, se cruzan los flags individuales con las features PRE
                        del merged. Si es None, se usan solo las columnas flag_/target del merged.
        post_variables: lista explícita de columnas POS a analizar. Si None, se auto-detectan.
        top_n:          features PRE a mostrar por variable POS (None = todas).
        random_state:   semilla para Información Mutua.
        active_versions: filtro de versiones activas.

    Retorna dict con:
        "per_flag_linkage": ranking de features PRE por cada flag POS y versión.
        "flag_summary":     resumen de señal por flag (max MI, n features informativas,
                            top 3 features predictoras, prevalencia del flag).
    """
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    if not merged_files:
        raise FileNotFoundError(f"No se encontraron datasets en {merged_dir}")

    active_set = set(active_versions) if active_versions is not None else None
    if active_set is not None:
        merged_files = [p for p in merged_files if p.stem.replace("OPERA_COMPLETO_", "") in active_set]
        if not merged_files:
            raise FileNotFoundError(f"No hay datasets para las versiones activas: {sorted(active_set)}")

    # Cargar mapa de archivos OPERA_POS si se provee pos_dir
    pos_map: dict[str, Path] = {}
    if pos_dir is not None:
        pos_dir = Path(pos_dir)
        for p in sorted(pos_dir.glob("OPERA_POS_*.xlsx")):
            ver = p.stem.replace("OPERA_POS_", "")
            if active_set is None or ver in active_set:
                pos_map[ver] = p

    all_linkage_rows: list[pd.DataFrame] = []

    for path in merged_files:
        version = path.stem.replace("OPERA_COMPLETO_", "")
        df_merged = pd.read_excel(path)

        if "Edad" in df_merged.columns:
            df_merged = df_merged[df_merged["Edad"] >= 18].reset_index(drop=True)

        # --- Features preoperatorias: columnas numéricas del merged sin flags ni target ---
        pre_cols = []
        known_pos = {c for c in df_merged.columns if c.startswith("flag_") or c == "target"}
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
            print(f"  [{version}] Sin features preoperatorias válidas — omitido.")
            continue

        X_pre_base = df_merged[pre_cols].apply(pd.to_numeric, errors="coerce")

        # --- Variables posoperatorias: flags individuales del OPERA_POS + target del merged ---
        if pos_dir is not None and version in pos_map:
            df_pos = pd.read_excel(pos_map[version])

            # Identificar columna de ID en cada dataset (puede tener nombres distintos)
            # Buscar por prefijo "Documento PMD" para tolerar variantes como
            # "Documento PMD (valoración preanestésica)"
            id_merged = next((c for c in df_merged.columns if c.startswith("Documento PMD") or c == "Documento_PMD"), None)
            id_pos = next((c for c in df_pos.columns if c.startswith("Documento PMD") or c == "Documento_PMD"), None)

            if id_merged is None or id_pos is None:
                print(f"  [{version}] No se encontró columna ID para cruzar PRE y POS — usando solo merged.")
                df_full = df_merged
            else:
                flag_cols_pos = [c for c in df_pos.columns if c.startswith("flag_")]
                df_flags = df_pos[[id_pos] + flag_cols_pos].copy()
                # Normalizar nombre del ID para hacer el merge
                df_flags = df_flags.rename(columns={id_pos: id_merged})
                df_full = df_merged.merge(df_flags, on=id_merged, how="inner", suffixes=("", "_pos"))
                df_full = df_full.reset_index(drop=True)
                # Recalcular X_pre con filas resultantes del merge
                X_pre_base = df_full[pre_cols].apply(pd.to_numeric, errors="coerce")
        else:
            df_full = df_merged

        # Determinar columnas POS a analizar
        if post_variables is not None:
            pos_cols = [c for c in post_variables if c in df_full.columns]
        else:
            pos_cols = sorted(
                [c for c in df_full.columns if c.startswith("flag_")] +
                (["target"] if "target" in df_full.columns else [])
            )

        if not pos_cols:
            print(f"  [{version}] Sin columnas posoperatorias — omitido.")
            continue

        X_pre = X_pre_base.fillna(-1)
        print(f"\n[{version}] Features PRE: {len(pre_cols)} | Variables POS: {len(pos_cols)}")

        for pos_var in pos_cols:
            y_post = pd.to_numeric(df_full[pos_var], errors="coerce").dropna().astype(int)
            if y_post.nunique() < 2:
                continue
            X_aligned = X_pre.loc[y_post.index]
            prevalence = float(y_post.mean())

            linkage = _compute_linkage_for_post_variable(
                X_aligned, y_post, version, pos_var, top_n=top_n, random_state=random_state
            )
            if not linkage.empty:
                linkage["Prevalencia_Post"] = round(prevalence, 4)
                all_linkage_rows.append(linkage)

            informative = int((linkage["Mutual_Information"] > 0.01).sum()) if not linkage.empty else 0
            max_mi = linkage["Mutual_Information"].max() if not linkage.empty else 0.0
            print(f"  {pos_var:<45} prevalencia={prevalence:.1%}  max_MI={max_mi:.4f}  n_informativas={informative}")

    if not all_linkage_rows:
        raise ValueError("No se pudo calcular ningún linkage PRE→POS.")

    df_linkage = pd.concat(all_linkage_rows, ignore_index=True)

    # --- Resumen por flag: señal máxima, features más predictoras ---
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

    linkage_path = output_dir / "pre_post_linkage_per_flag.csv"
    summary_path = output_dir / "pre_post_linkage_summary.csv"
    df_linkage.to_csv(linkage_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✓ Exportado: {linkage_path}")
    print(f"✓ Exportado: {summary_path}")

    return {"per_flag_linkage": df_linkage, "flag_summary": df_summary}
