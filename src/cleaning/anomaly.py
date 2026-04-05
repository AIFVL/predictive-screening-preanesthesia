# src/cleaning/anomaly.py
"""
Detección de anomalías post-limpieza con IsolationForest.
Migrado fielmente desde notebooks/2_data_cleaning.ipynb (cells 51-52).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("cleaning.anomaly")

PERSONAL_COLS = [
    "Documento PMD",
    "Número de episodio",
    "Nombres y Apellidos",
    "Tipo de identificación",
    "Número de identificación",
]


def detectar_anomalias_y_exportar(
    df: pd.DataFrame,
    metodo: str = "isolation_forest",
    contamination: float = 0.5,
    ruta_exportacion: str | Path = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detecta anomalías en el dataset y exporta el resultado con columna 'es_anomalia'.
    Replica fielmente detectar_anomalias_y_exportar() del notebook.

    Parámetros:
        df: DataFrame limpio (tras clean_preop + enrich_preop)
        metodo: 'isolation_forest' (único soportado en este módulo)
        contamination: proporción esperada de anomalías (0.0 - 0.5)
        ruta_exportacion: path CSV de salida (None = no exporta)
        random_state: semilla para reproducibilidad

    Retorna:
        DataFrame con columna 'es_anomalia' (1=anomalía, 0=normal) y 'Documento PMD'.
    """
    df_result = df.copy()

    # Guardar Documento PMD para reintegrarlo al final
    documento_pmd = None
    if "Documento PMD" in df_result.columns:
        documento_pmd = df_result["Documento PMD"].reset_index(drop=True)

    # Eliminar columnas personales + seleccionar solo numéricas
    cols_to_drop = [c for c in PERSONAL_COLS if c in df_result.columns]
    df_result = df_result.drop(columns=cols_to_drop)
    df_result = df_result.select_dtypes(include=[np.number])

    logger.info(f"Filas a analizar: {len(df_result):,}")

    X = df_result.copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if metodo == "isolation_forest":
        detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        anomalias = detector.fit_predict(X_scaled)
        df_result["es_anomalia"] = (anomalias == -1).astype(int)
    else:
        raise ValueError(f"Método '{metodo}' no reconocido. Use 'isolation_forest'.")

    total_anomalias = df_result["es_anomalia"].sum()
    pct = total_anomalias / len(df_result) * 100
    logger.info(f"Anomalías detectadas: {total_anomalias:,} ({pct:.2f}%)")

    if documento_pmd is not None:
        df_result["Documento PMD"] = documento_pmd.values

    if ruta_exportacion is not None:
        ruta_exportacion = Path(ruta_exportacion)
        ruta_exportacion.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(ruta_exportacion, index=False)
        logger.info(f"Exportado: {ruta_exportacion}")

    return df_result
