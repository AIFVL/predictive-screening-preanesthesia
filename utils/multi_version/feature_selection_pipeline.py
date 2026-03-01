from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from utils.feature_engineering import (
    ENCODING_FIX_MAP,
    resolve_feature_columns,
    sanitize_features_for_subset,
)


def get_version_name(path: Path) -> str:
    return path.stem.replace("OPERA_COMPLETO_", "")


def rank_features_for_version(
    df_input: pd.DataFrame,
    version_name: str,
    selected_names: list[str],
    features_meta: pd.DataFrame,
    top_n: int = 80,
    random_state: int = 42,
):
    df = df_input.copy()
    if "target" not in df.columns:
        raise ValueError(f"[{version_name}] No existe columna 'target'")

    if "Edad" in df.columns:
        df = df[df["Edad"] >= 18].reset_index(drop=True)

    df = df.rename(columns=ENCODING_FIX_MAP)

    feature_columns, missing = resolve_feature_columns(selected_names, df.columns, features_meta)
    feature_columns = [ENCODING_FIX_MAP.get(c, c) for c in feature_columns]
    feature_columns, dropped = sanitize_features_for_subset(df, feature_columns)

    X = df[feature_columns].copy()
    y = df["target"].astype(int).copy()
    X_filled = X.fillna(-1)

    mi_scores = mutual_info_classif(X_filled, y, random_state=random_state, n_neighbors=5)
    mi_df = pd.DataFrame({"Variable": X_filled.columns, "MI_Score": mi_scores})

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_filled, y)
    rf_df = pd.DataFrame({"Variable": X_filled.columns, "RF_Importance": rf.feature_importances_})

    ranking = mi_df.merge(rf_df, on="Variable", how="outer").fillna(0)

    scaler = MinMaxScaler()
    ranking["MI_Norm"] = scaler.fit_transform(ranking[["MI_Score"]])
    ranking["RF_Norm"] = scaler.fit_transform(ranking[["RF_Importance"]])
    ranking["Combined_Score"] = ranking[["MI_Norm", "RF_Norm"]].mean(axis=1)
    ranking = ranking.sort_values("Combined_Score", ascending=False).reset_index(drop=True)

    selected = ranking.head(top_n).copy()
    selected["Selected"] = True

    metadata = {
        "version": version_name,
        "rows": len(df),
        "target_1_pct": round(y.mean() * 100, 2),
        "candidate_features": len(feature_columns),
        "selected_features": len(selected),
        "missing_features": len(missing),
        "dropped_constant": len(dropped),
    }
    return ranking, selected, metadata


def run_feature_selection_pipeline(
    merged_dir: Path,
    features_meta_path: Path,
    output_dir: Path,
    top_n: int = 80,
    random_state: int = 42,
    active_versions: list[str] | None = None,
):
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    if not merged_files:
        raise FileNotFoundError("No se encontraron datasets en data/merged_versions")

    ignored_versions = []
    if active_versions is not None:
        active_set = set(active_versions)
        filtered = []
        for path in merged_files:
            version = get_version_name(path)
            if version in active_set:
                filtered.append(path)
            else:
                ignored_versions.append(version)

        merged_files = filtered
        if not merged_files:
            raise FileNotFoundError(
                "No se encontraron datasets mergeados para las versiones activas: "
                f"{sorted(active_set)}"
            )

    features_meta = pd.read_csv(features_meta_path)
    selected_names = features_meta["Variable"].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)

    rankings_by_version = {}
    selected_by_version = {}
    version_summaries = []

    for path in merged_files:
        version = get_version_name(path)
        df_version = pd.read_excel(path)

        ranking, selected, metadata = rank_features_for_version(
            df_version,
            version_name=version,
            selected_names=selected_names,
            features_meta=features_meta,
            top_n=top_n,
            random_state=random_state,
        )

        rankings_by_version[version] = ranking
        selected_by_version[version] = selected
        version_summaries.append(metadata)

    df_version_summary = pd.DataFrame(version_summaries).sort_values("version").reset_index(drop=True)

    all_selected = []
    for version, selected_df in selected_by_version.items():
        tmp = selected_df[["Variable", "Combined_Score"]].copy()
        tmp["version"] = version
        all_selected.append(tmp)

    df_all_selected = pd.concat(all_selected, ignore_index=True)

    feature_frequency = (
        df_all_selected.groupby("Variable")["version"]
        .nunique()
        .reset_index(name="n_versions_selected")
        .sort_values(["n_versions_selected", "Variable"], ascending=[False, True])
        .reset_index(drop=True)
    )

    selection_matrix = pd.crosstab(
        df_all_selected["Variable"],
        df_all_selected["version"],
        values=1,
        aggfunc="sum",
        dropna=False,
    ).fillna(0).astype(int)

    export_registry = []
    for version in sorted(rankings_by_version.keys()):
        ranking_path = output_dir / f"{version}_variable_ranking.csv"
        selected_path = output_dir / f"{version}_variables_selected.csv"
        selected_txt_path = output_dir / f"{version}_selected_features_list.txt"

        rankings_by_version[version].to_csv(ranking_path, index=False)
        selected_by_version[version].to_csv(selected_path, index=False)

        with open(selected_txt_path, "w", encoding="utf-8") as f:
            for feature in selected_by_version[version]["Variable"].tolist():
                f.write(f"{feature}\n")

        export_registry.append(
            {
                "version": version,
                "ranking_path": str(ranking_path),
                "selected_path": str(selected_path),
                "selected_txt_path": str(selected_txt_path),
                "n_selected": len(selected_by_version[version]),
            }
        )

    df_export_registry = pd.DataFrame(export_registry).sort_values("version").reset_index(drop=True)

    summary_path = output_dir / "summary_versions.csv"
    frequency_path = output_dir / "feature_frequency_across_versions.csv"
    matrix_path = output_dir / "selection_matrix_across_versions.csv"
    registry_path = output_dir / "exports_registry_versions.csv"

    df_version_summary.to_csv(summary_path, index=False)
    feature_frequency.to_csv(frequency_path, index=False)
    selection_matrix.to_csv(matrix_path, index=True)
    df_export_registry.to_csv(registry_path, index=False)

    min_versions = max(1, int(np.ceil(len(df_version_summary) * 0.5)))
    consensus_features = feature_frequency[
        feature_frequency["n_versions_selected"] >= min_versions
    ]["Variable"].tolist()

    consensus_path = output_dir / "consensus_selected_features_list.txt"
    with open(consensus_path, "w", encoding="utf-8") as f:
        for feature in consensus_features:
            f.write(f"{feature}\n")

    versions_df = pd.DataFrame(
        [{"version": get_version_name(path), "file": path.name} for path in merged_files]
    ).sort_values("version")

    return {
        "versions": versions_df,
        "ignored_versions": sorted(set(ignored_versions)),
        "rankings_by_version": rankings_by_version,
        "selected_by_version": selected_by_version,
        "summary": df_version_summary,
        "feature_frequency": feature_frequency,
        "selection_matrix": selection_matrix,
        "export_registry": df_export_registry,
        "consensus_features": consensus_features,
        "summary_path": summary_path,
        "frequency_path": frequency_path,
        "matrix_path": matrix_path,
        "registry_path": registry_path,
        "consensus_path": consensus_path,
    }
