from __future__ import annotations

from pathlib import Path

import pandas as pd

from api.domain.manifest import ModelManifest
from api.services.preprocessor import apply_imputation

from src.cleaning.cleaner import clean_preop
from src.cleaning.enrichment import enrich_preop


# Normalización de campos categóricos estrictos: el OrdinalEncoder en el pipeline
# rechaza valores desconocidos, así que mapeamos variantes comunes al valor esperado.
_STRICT_NORMS: dict[str, dict[str, str]] = {
    "Sexo": {
        "masculino": "M", "male": "M", "hombre": "M", "m": "M",
        "femenino": "F", "female": "F", "mujer": "F", "f": "F",
    },
    "Atención": {
        "urgencia": "Urgencia", "urgente": "Urgencia", "urgency": "Urgencia",
        "electivo": "Electivo", "elective": "Electivo", "programado": "Electivo",
        "urgencia programada": "Urgencia Programada",
    },
    "Cuello Móvil": {
        "normal": "Normal", "si": "Normal", "sí": "Normal", "yes": "Normal",
        "reducido": "Reducido", "fijo": "Fijo", "no": "Fijo",
    },
    "Apertura Oral": {
        "> 3 cm": "> 3 cm", "normal": "> 3 cm", "si": "> 3 cm", "sí": "> 3 cm",
        "< 3 cm": "< 3 cm", "no": "< 3 cm", "disminuida": "< 3 cm", "limitada": "< 3 cm",
    },
    "Arritmia": {
        "si": "Si", "sí": "Si", "yes": "Si", "1": "Si", "true": "Si",
        "no": "No", "0": "No", "false": "No",
    },
}


def _normalize_strict(row: dict) -> dict:
    """Mapea variantes comunes a los valores exactos que el OrdinalEncoder espera."""
    for col, mapping in _STRICT_NORMS.items():
        if col in row and isinstance(row[col], str):
            row[col] = mapping.get(row[col].strip().lower(), row[col])
    return row


def preprocess_raw(
    patient: dict,
    manifest: ModelManifest,
    cache_dir: Path,
    cleaning_cfg: dict,
) -> pd.DataFrame:
    """
    Convierte los datos clínicos crudos al espacio de features del modelo.

    Recibe `patient` con claves en el nombre exacto de columna del dataset
    (español, con tildes y espacios), tal como los devuelve el manifest en
    `raw_input_schema`.

    Flujo:
      1. Aplica correcciones estructurales que el pipeline de limpieza necesita.
      2. Llama clean_preop() (limpieza determinista).
      3. Llama enrich_preop() (ICD coding + severidad BART-MNLI).
      4. Reindexar e imputar según el manifest.
    """
    def _is_empty(v) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and v != v:  # NaN
            return True
        if isinstance(v, str) and v.strip().lower() in ("nan", ""):
            return True
        return False

    row: dict = {k: v for k, v in patient.items() if not _is_empty(v)}
    row = _normalize_strict(row)

    # Garantizar que todas las columnas del schema raw estén presentes en el DataFrame
    # (con None) para que el cleaner no falle con KeyError en columnas opcionales ausentes.
    if manifest.raw_input_schema:
        for field_def in manifest.raw_input_schema:
            if field_def["name"] not in row:
                row[field_def["name"]] = None

    # clean_fecha_hora() requiere Fecha como datetime, no string
    if "Fecha" in row:
        row["Fecha"] = pd.to_datetime(row["Fecha"])

    # clean_sexo_atencion() necesita ambas columnas para dispararse;
    # sin Atención, Sexo no se codifica y Sexo_encoded quedaría ausente
    if "Sexo" in row and "Atención" not in row:
        row["Atención"] = "Electivo"

    # clean_categorical_physical_exam() necesita "Estado Nutricional" para dispararse,
    # pero lo recalcula internamente desde IMC — el valor inicial no importa
    if "IMC" in row and "Estado Nutricional" not in row:
        row["Estado Nutricional"] = None

    # clean_alergenos() siempre intenta dropear "Alergias"
    if "Alérgeno" in row:
        row["Alergias"] = None

    # clean_dx_proc() accede a Procedimiento propuesto y Antecedentes quirúrgicos
    # incluso si no fueron provistos; establecer defaults vacíos evita KeyError
    if "Dx Preoperatorio" in row:
        if "Procedimiento propuesto" not in row:
            row["Procedimiento propuesto"] = ""
        if "Antecedentes quirúrgicos" not in row:
            row["Antecedentes quirúrgicos"] = ""

    # Guardar columnas que clean_antecedentes_anestesia() dropea pero enrich_preop() necesita
    _saved_for_enrichment = {
        col: row[col]
        for col in ("Antecedentes quirúrgicos", "Dx Preoperatorio", "Procedimiento propuesto")
        if col in row
    }

    df = pd.DataFrame([row])
    df = clean_preop(df, cleaning_cfg)

    # Restaurar columnas de enriquecimiento que el pipeline de limpieza pudo haber eliminado
    for col, val in _saved_for_enrichment.items():
        if col not in df.columns:
            df[col] = val

    df = enrich_preop(df, cache_dir)

    df = df.reindex(columns=manifest.feature_names)
    df = apply_imputation(df, manifest)
    return df
