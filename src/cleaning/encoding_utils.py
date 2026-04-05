# src/cleaning/encoding_utils.py
"""
Utilidades de codificación de variables categóricas.
Migradas fielmente desde notebooks/2_data_cleaning.ipynb (cell 13).
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
)


def impute(df: pd.DataFrame, columns: list, value="negativo") -> pd.DataFrame:
    """Imputa valores faltantes con un valor personalizado."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(value)
    return df_copy


def encode_multilabel(df: pd.DataFrame, column: str, delimiter: str = ",") -> pd.DataFrame:
    """Codifica variables multilabel — produce columnas {column}_{clase}."""
    df_copy = df.copy()
    values = df_copy[column].apply(
        lambda x: [i.strip() for i in str(x).split(delimiter) if i.strip()]
    )
    encoder = MultiLabelBinarizer()
    encoded = pd.DataFrame(
        encoder.fit_transform(values),
        columns=[f"{column}_{cls}" for cls in encoder.classes_],
        index=df_copy.index,
    )
    return pd.concat([df_copy, encoded], axis=1)


def encode_onehot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Codifica una variable categórica con one-hot encoding."""
    df_copy = df.copy()
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    encoded_array = encoder.fit_transform(df_copy[[column]])
    encoded = pd.DataFrame(
        encoded_array,
        columns=[f"{column}_{cls}" for cls in encoder.categories_[0]],
        index=df_copy.index,
    )
    return pd.concat([df_copy, encoded], axis=1)


def encode_label_with_classes(
    df: pd.DataFrame, column: str, classes: list
) -> pd.DataFrame:
    """
    Codifica una columna con OrdinalEncoder usando clases definidas explícitamente.
    Produce columna {column}_encoded.
    """
    df_copy = df.copy()
    encoder = OrdinalEncoder(categories=[classes])
    df_copy[f"{column}_encoded"] = encoder.fit_transform(df_copy[[column]])
    return df_copy
