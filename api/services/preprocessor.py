from __future__ import annotations

import pandas as pd

from api.domain.manifest import ModelManifest


def features_dict_to_dataframe(
    features: dict | list[dict],
    manifest: ModelManifest,
) -> pd.DataFrame:
    if isinstance(features, dict):
        rows = [features]
    else:
        rows = list(features)

    df = pd.DataFrame(rows)
    # Reordena y agrega columnas faltantes como NaN.
    df = df.reindex(columns=manifest.feature_names)
    return df


def apply_imputation(df: pd.DataFrame, manifest: ModelManifest) -> pd.DataFrame:
    """Aplica la estrategia declarada en `manifest.imputation`."""
    strategy = manifest.imputation.get("strategy", "fill_constant")
    df = df.apply(pd.to_numeric, errors="coerce")

    if strategy == "fill_constant":
        fill_value = manifest.imputation.get("value", -1)
        return df.fillna(fill_value)
    if strategy == "fill_median":
        medians = pd.Series(manifest.feature_medians)
        return df.fillna(medians)
    raise ValueError(f"Estrategia de imputación desconocida: {strategy!r}")


def preprocess(features: dict | list[dict], manifest: ModelManifest) -> pd.DataFrame:
    """Pipeline completo: dict → DataFrame ordenado → imputado → numérico."""
    df = features_dict_to_dataframe(features, manifest)
    return apply_imputation(df, manifest)
