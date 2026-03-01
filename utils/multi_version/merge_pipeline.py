from pathlib import Path
import pandas as pd


def infer_version_name(file_path: Path) -> str:
    stem = file_path.stem
    return stem.replace("OPERA_POS_", "") if stem.startswith("OPERA_POS_") else stem


def run_merge_pipeline(
    data_dir: Path,
    merged_dir: Path,
    active_target_files: list[str] | None = None,
):
    preop_path = data_dir / "OPERA_PRE.xlsx"
    df_preop = pd.read_excel(preop_path)

    target_files = sorted(data_dir.glob("OPERA_POS*.xlsx"))
    if not target_files:
        raise FileNotFoundError("No se encontraron archivos OPERA_POS*.xlsx en ../data")

    ignored_target_files = []
    if active_target_files is not None:
        active_set = set(active_target_files)
        filtered_files = [path for path in target_files if path.name in active_set]
        ignored_target_files = [path.name for path in target_files if path.name not in active_set]
        target_files = filtered_files

        if not target_files:
            raise FileNotFoundError(
                "No se encontraron archivos target activos. "
                f"Esperados: {sorted(active_set)}"
            )

    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_by_version = {}
    merge_summary = []
    exports = []

    for file_path in target_files:
        df_target = pd.read_excel(file_path)

        if "Documento PMD (valoración preanestésica)" in df_target.columns:
            df_target = df_target.rename(columns={"Documento PMD (valoración preanestésica)": "Documento PMD"})

        required_cols = {"Documento PMD", "target"}
        missing_cols = required_cols - set(df_target.columns)
        if missing_cols:
            raise ValueError(f"{file_path.name} no contiene columnas requeridas: {missing_cols}")

        version = infer_version_name(file_path)

        df_merged = pd.merge(
            df_preop,
            df_target[["Documento PMD", "target"]],
            on="Documento PMD",
            how="inner",
        )

        merged_by_version[version] = df_merged

        target_rate = df_merged["target"].mean() * 100
        merge_summary.append(
            {
                "version": version,
                "target_file": file_path.name,
                "rows": len(df_merged),
                "cols": df_merged.shape[1],
                "target_0": int((df_merged["target"] == 0).sum()),
                "target_1": int((df_merged["target"] == 1).sum()),
                "target_1_pct": round(target_rate, 2),
            }
        )

        output_path = merged_dir / f"OPERA_COMPLETO_{version}.xlsx"
        df_merged.to_excel(output_path, index=False)
        exports.append({"version": version, "path": str(output_path), "rows": len(df_merged)})

    df_merge_summary = pd.DataFrame(merge_summary).sort_values("version").reset_index(drop=True)
    df_exports = pd.DataFrame(exports).sort_values("version").reset_index(drop=True)

    summary_path = merged_dir / "merge_summary_versions.csv"
    exports_path = merged_dir / "merged_exports_versions.csv"

    df_merge_summary.to_csv(summary_path, index=False)
    df_exports.to_csv(exports_path, index=False)

    return {
        "preop_path": preop_path,
        "preop_shape": df_preop.shape,
        "target_files": [str(path) for path in target_files],
        "ignored_target_files": ignored_target_files,
        "merged_by_version": merged_by_version,
        "merge_summary": df_merge_summary,
        "exports": df_exports,
        "summary_path": summary_path,
        "exports_path": exports_path,
    }
