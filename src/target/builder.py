import pandas as pd

from .constants import EXCLUDED_FLAGS, RELEVANT_FLAGS


def build_target(
    df: pd.DataFrame,
    threshold: int = 1,
    flags_to_use: list[str] | None = None,
    relevant_flags: list[str] | None = None,
    excluded_flags: list[str] | None = None,
    apply_cancel_non_medico_rule: bool = True,
    version_name: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    df_result = df.copy()

    if flags_to_use is None:
        flags_to_use = relevant_flags
    flags_to_use = flags_to_use or RELEVANT_FLAGS
    excluded_flags = excluded_flags or EXCLUDED_FLAGS

    missing_relevant = [flag for flag in flags_to_use if flag not in df_result.columns]
    if missing_relevant:
        raise ValueError(
            "No se encontraron las siguientes columnas de flags relevantes en el DataFrame: "
            f"{missing_relevant}"
        )

    df_result["n_flags_relevant"] = df_result[flags_to_use].sum(axis=1)
    df_result["target"] = (df_result["n_flags_relevant"] >= threshold).astype(int)

    if (
        apply_cancel_non_medico_rule
        and "canceladas" in df_result.columns
        and "canceladas_por_medico" in df_result.columns
    ):
        mask_cancelada_no_medico = (df_result["canceladas"] == 1) & (
            df_result["canceladas_por_medico"] == 0
        )
        df_result.loc[mask_cancelada_no_medico, "target"] = 0

    if verbose:
        title_suffix = f" - {version_name}" if version_name else ""
        print(f"\n{'=' * 70}")
        print(f"VARIABLE OBJETIVO{title_suffix} (threshold ≥ {threshold} flags)")
        print(f"{'=' * 70}")
        print(f"\nFlags utilizados ({len(flags_to_use)}):")
        for flag in flags_to_use:
            n = df_result[flag].sum()
            print(f"  • {flag}: {n:,} ({n / len(df_result) * 100:.2f}%)")
        print(f"\nFlags excluidos ({len(excluded_flags)}):")
        for flag in excluded_flags:
            if flag in df_result.columns:
                n = df_result[flag].sum()
                print(f"  ✗ {flag}: {n:,} ({n / len(df_result) * 100:.2f}%)")
            else:
                print(f"  ✗ {flag}: (no presente en dataset)")
        print("\nDistribución de n_flags_relevant:")
        for k in sorted(df_result["n_flags_relevant"].unique()):
            n = (df_result["n_flags_relevant"] == k).sum()
            print(f"  {k} flags: {n:,} ({n / len(df_result) * 100:.2f}%)")
        print(f"\nDistribución del target (threshold ≥ {threshold}):")
        print(
            f"  Clase 0: {(df_result['target'] == 0).sum():,} "
            f"({(df_result['target'] == 0).mean() * 100:.2f}%)"
        )
        print(
            f"  Clase 1: {(df_result['target'] == 1).sum():,} "
            f"({(df_result['target'] == 1).mean() * 100:.2f}%)"
        )

    return df_result
