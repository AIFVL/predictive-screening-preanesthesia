from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


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

    return result.sort_values(["Combined_Score", "Mutual_Information"], ascending=False).head(top_n).reset_index(drop=True)
