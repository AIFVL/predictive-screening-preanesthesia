# src/cleaning/numeric_utils.py
"""
Utilidades de conversión y validación de variables numéricas con imputación por rangos.
Migradas fielmente desde notebooks/2_data_cleaning.ipynb (cells 11, 16).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def convert_to_numeric(
    df: pd.DataFrame, columns: list, errors: str = "coerce"
) -> pd.DataFrame:
    """Convierte columnas a tipo numérico."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors=errors)
    return df_copy


def validate_numeric_ranges_with_noise(
    df: pd.DataFrame,
    columns: list,
    ranges: dict,
    ruido_pct: float = 0.05,
    seed: int = None,
) -> pd.DataFrame:
    """
    Para cada columna, si el valor está fuera del rango (min, max) lo imputa con
    un valor aleatorio dentro del rango + ruido proporcional.
    Seed=42 para reproducibilidad exacta del notebook.
    """
    df_copy = df.copy()
    if seed is not None:
        np.random.seed(seed)

    for col in columns:
        if col not in df_copy.columns or col not in ranges:
            continue
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").astype(float)
        min_val, max_val = ranges[col]
        mask_fuera = ~df_copy[col].between(min_val, max_val) & df_copy[col].notna()
        n_imputar = mask_fuera.sum()
        if n_imputar > 0:
            rango = max_val - min_val
            ruido = np.random.uniform(-ruido_pct, ruido_pct, size=n_imputar)
            random_base = np.random.rand(n_imputar)
            imputados = (min_val + rango * random_base) * (1 + ruido)
            df_copy.loc[mask_fuera, col] = np.clip(imputados, min_val, max_val)

    return df_copy


def imputar_fuera_de_rango(
    df: pd.DataFrame,
    rangos_por_edad: list,
    edad_col: str = "Edad",
    ruido_pct: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Imputa peso, talla e IMC fuera de rango según grupo de edad con ruido proporcional.
    Migrado fielmente de notebook cell 16.
    """
    np.random.seed(seed)
    df_copy = df.copy()

    cols_monitoreo = ["Peso (Kg)", "Talla (cm)", "IMC", edad_col]
    for col in cols_monitoreo:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").astype(float)

    df_copy["recalculado_flag"] = False

    # Skip body-composition imputation if the required columns are absent
    _has_talla = "Talla (cm)" in df_copy.columns
    _has_peso = "Peso (Kg)" in df_copy.columns
    if not (_has_talla and _has_peso):
        return df_copy

    talla_m = df_copy["Talla (cm)"] / 100
    imc_calculado = np.where(
        talla_m > 0, df_copy["Peso (Kg)"] / (talla_m ** 2), np.nan
    )

    for grupo in rangos_por_edad:
        edad_mask = (
            (df_copy[edad_col] >= grupo["edad_min"])
            & (df_copy[edad_col] <= grupo["edad_max"])
            & df_copy[edad_col].notna()
        )
        if not edad_mask.any():
            continue

        peso_fuera = (
            (df_copy["Peso (Kg)"] < grupo["peso"][0])
            | (df_copy["Peso (Kg)"] > grupo["peso"][1])
            | df_copy["Peso (Kg)"].isna()
        )
        talla_fuera = (
            (df_copy["Talla (cm)"] < grupo["talla"][0])
            | (df_copy["Talla (cm)"] > grupo["talla"][1])
            | df_copy["Talla (cm)"].isna()
        )
        imc_serie = pd.Series(imc_calculado, index=df_copy.index)
        imc_fuera = (
            (imc_serie < grupo["imc"][0])
            | (imc_serie > grupo["imc"][1])
            | imc_serie.isna()
        )

        mask_imputar = edad_mask & (peso_fuera | talla_fuera | imc_fuera)
        n_impute = mask_imputar.sum()

        if n_impute > 0:
            peso_mean = np.mean(grupo["peso"])
            talla_mean = np.mean(grupo["talla"])
            imc_mean = np.mean(grupo["imc"])

            df_copy.loc[mask_imputar, "Peso (Kg)"] = peso_mean * (
                1 + np.random.uniform(-ruido_pct, ruido_pct, n_impute)
            )
            df_copy.loc[mask_imputar, "Talla (cm)"] = talla_mean * (
                1 + np.random.uniform(-ruido_pct, ruido_pct, n_impute)
            )
            df_copy.loc[mask_imputar, "IMC"] = imc_mean * (
                1 + np.random.uniform(-ruido_pct, ruido_pct, n_impute)
            )
            df_copy.loc[mask_imputar, "recalculado_flag"] = True

        mask_recalc = edad_mask & ~mask_imputar & (df_copy["Talla (cm)"] > 0)
        if mask_recalc.any():
            df_copy.loc[mask_recalc, "IMC"] = df_copy.loc[
                mask_recalc, "Peso (Kg)"
            ] / ((df_copy.loc[mask_recalc, "Talla (cm)"] / 100) ** 2)

    return df_copy
