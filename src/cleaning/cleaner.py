# src/cleaning/cleaner.py
"""
Pipeline de limpieza determinista del dataset preoperatorio OPERA.
Migrado fielmente desde notebooks/2_data_cleaning.ipynb cells 8-36.

Todas las funciones clean_* son deterministas (sin APIs externas).
Las funciones que requieren APIs externas (clean_medicamentos, clean_dx_proc)
están en src/cleaning/enrichment.py.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.cleaning.encoding_utils import (
    encode_label_with_classes,
    encode_multilabel,
    encode_onehot,
    impute,
)
from src.cleaning.numeric_utils import (
    convert_to_numeric,
    imputar_fuera_de_rango,
    validate_numeric_ranges_with_noise,
)
from src.cleaning.text_utils import (
    create_multi_label,
    normalize_tokens,
    resolve_contradictory_term,
    standardize_text,
)
from src.utils.logger import get_logger

logger = get_logger("cleaning.cleaner")


# ── Parsers de laboratorio ────────────────────────────────────────────────────

def _to_float(s, default=None):
    raw = ("" if pd.isna(s) else str(s)).strip()
    tmp = raw.replace(",", ".")
    tmp = re.sub(r"[<>≈~]", "", tmp)
    tmp = re.sub(r"\s+", " ", tmp).strip()
    try:
        return float(tmp)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+", tmp)
        return float(m.group()) if m else default


def _pt_inr_from_text(
    s,
    pt_range=(7, 1000),
    inr_range=(0.7, 8),
    pt_control=11.5,
    isi=1.0,
):
    if s is None:
        return None
    txt = str(s).strip().replace(",", ".")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", txt)]

    def is_pt(x):
        return pt_range[0] <= x <= pt_range[1]

    def is_inr(x):
        return inr_range[0] <= x <= inr_range[1]

    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if is_pt(a) and is_inr(b):
            return b
        if is_pt(b) and is_inr(a):
            return a
        if is_inr(b):
            return b
        if is_inr(a):
            return a
        if is_pt(a):
            return (a / pt_control) ** isi
        if is_pt(b):
            return (b / pt_control) ** isi
        return None
    elif len(nums) == 1:
        x = nums[0]
        if is_inr(x):
            return x
        if is_pt(x):
            return (x / pt_control) ** isi
        return None
    return None


def _to_float_percent(s, default=None):
    if s is None:
        return default
    txt = str(s).strip().replace("%", "")
    value = _to_float(txt)
    if value is None:
        return default
    return value if 0 < value < 100 else default


def _to_10e3_per_uL_from_count(s):
    if s is None:
        return None
    raw = str(s).strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
        normalized = raw.replace(".", "")
        try:
            return float(normalized) / 1000.0
        except Exception:
            return _to_float(raw)
    tmp = raw.replace(",", ".")
    if re.fullmatch(r"\d+\.\d{3}$", tmp):
        try:
            return float(tmp.replace(".", "")) / 1000.0
        except Exception:
            pass
    val = _to_float(tmp)
    if val is None:
        return None
    return val / 1000.0 if val >= 1000 else val


def _to_pdo(result):
    states = {
        0: ["OK", "NORMAL", "NO INFECCION", "NO INFECCIÓN", "SIN INFECCION",
            "SIN INFECCIÓN", "NEGATIVO"],
        1: ["UROCULTIVO"],
    }
    for state, values in states.items():
        if result in values:
            return state
    return 0


EXAM_PARSERS = {
    "Hemoglobina(g/dl)":    lambda r: _to_float(r),
    "PT (INR)":             lambda r: _pt_inr_from_text(r) if r is not None else None,
    "Glicemia (mg/dl)":     lambda r: _to_float(r),
    "Hematocrito (%)":      lambda r: _to_float_percent(r),
    "Plaquetas (ml)":       lambda r: _to_10e3_per_uL_from_count(r),
    "Leucocitos (ml)":      lambda r: _to_10e3_per_uL_from_count(r),
    "PTT":                  lambda r: (
        _to_float(str(r).split("/")[0])
        if r is not None and "/" in str(r)
        else _to_float(r)
    ),
    "K (meq/L)":            lambda r: _to_float(r),
    "Na (meq/L)":           lambda r: _to_float(r),
    "P de O":               lambda r: _to_pdo(r),
}


# ── clean_fecha_hora ──────────────────────────────────────────────────────────

def _parse_hora(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return pd.to_datetime(str(s)).time()
    except Exception:
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return pd.to_datetime(str(s), format=fmt, errors="raise").time()
            except Exception:
                continue
    return pd.NaT


def clean_fecha_hora(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc["anio"] = dfc["Fecha"].dt.year
    dfc["mes"] = dfc["Fecha"].dt.month
    dfc["dia_semana"] = dfc["Fecha"].dt.dayofweek
    dfc["Hora"] = dfc["Hora"].apply(_parse_hora)
    dfc["Hora_decimal"] = dfc["Hora"].apply(
        lambda x: x.hour + x.minute / 60 if pd.notna(x) else np.nan
    )
    dfc = dfc.drop(columns=["Fecha", "Hora"])
    return dfc


# ── clean_sexo_atencion ───────────────────────────────────────────────────────

def clean_sexo_atencion(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = encode_label_with_classes(dfc, "Sexo", classes=["M", "F"])
    dfc = encode_label_with_classes(
        dfc, "Atención", classes=["Electivo", "Urgencia", "Urgencia Programada"]
    )
    dfc = dfc.drop(columns=["Sexo", "Atención"])
    return dfc


# ── clean_variables_fisiologicas ──────────────────────────────────────────────

RANGOS_POR_EDAD = [
    {"edad_min": 0,  "edad_max": 2,   "peso": (2, 20),   "talla": (50, 95),   "imc": (12, 20)},
    {"edad_min": 3,  "edad_max": 10,  "peso": (10, 50),  "talla": (80, 140),  "imc": (13, 22)},
    {"edad_min": 11, "edad_max": 17,  "peso": (30, 90),  "talla": (120, 180), "imc": (14, 28)},
    {"edad_min": 18, "edad_max": 110, "peso": (40, 300), "talla": (140, 230), "imc": (15, 40)},
]

VITAL_RANGES = {
    "Tensión Arterial Sistólica (mm/Hg)":  (60, 260),
    "Tensión Arterial Diastólica (mm/Hg)": (30, 160),
    "Frecuencia Respiratoria":             (10, 60),
    "Temperatura":                         (26, 43),
    "Edad":                                (0, 110),
}


def clean_variables_fisiologicas(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    num_cols = [
        "Edad", "Peso (Kg)", "Talla (cm)", "IMC",
        "Tensión Arterial Sistólica (mm/Hg)", "Tensión Arterial Diastólica (mm/Hg)",
        "Tensión Arterial Media (mm/Hg)", "Frecuencia Respiratoria", "Temperatura",
    ]
    dfc = convert_to_numeric(dfc, num_cols, errors="coerce")
    dfc = validate_numeric_ranges_with_noise(
        dfc, list(VITAL_RANGES.keys()), VITAL_RANGES, ruido_pct=0.03, seed=42
    )
    dfc = imputar_fuera_de_rango(dfc, RANGOS_POR_EDAD)

    # Recalcular TAM si está fuera de rango (solo si las columnas existen)
    _tam_cols = [
        "Tensión Arterial Sistólica (mm/Hg)",
        "Tensión Arterial Diastólica (mm/Hg)",
        "Tensión Arterial Media (mm/Hg)",
    ]
    if all(c in dfc.columns for c in _tam_cols):
        tam = (
            dfc["Tensión Arterial Sistólica (mm/Hg)"]
            + 2 * dfc["Tensión Arterial Diastólica (mm/Hg)"]
        ) / 3
        ok = dfc["Tensión Arterial Media (mm/Hg)"].between(40, 200)
        dfc["Tensión Arterial Media (mm/Hg)"] = np.where(
            ok, dfc["Tensión Arterial Media (mm/Hg)"], tam
        )
    return dfc


# ── clean_grupo_rh_y_nyha ─────────────────────────────────────────────────────

def _nyha_to_int(x):
    return str(x).replace("I", "0").replace("II", "1").replace("III", "2").replace("IV", "3")


def clean_grupo_rh_y_nyha(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = encode_onehot(dfc, "Grupo Sanguíneo")
    dfc = encode_onehot(dfc, "RH")
    dfc = dfc.rename(columns={"Clase Funcional (NYHA) ": "Clase Funcional (NYHA)"})
    dfc["NYHA_num"] = dfc["Clase Funcional (NYHA)"].apply(_nyha_to_int)
    dfc = dfc.drop(columns=["Grupo Sanguíneo", "RH", "Clase Funcional (NYHA)"])
    return dfc


# ── clean_antecedentes ────────────────────────────────────────────────────────

ANTECEDENTES_COLS = [
    "Antecedente neurológicos",
    "Antecedente respiratorios",
    "Antecedentes cardiovasculares",
    "Antecedente hematológicos ",
    "Antecedente endocrinológicos",
    "Antecedente renales",
    "Antecedente gastrointestinales",
]


def clean_antecedente(df: pd.DataFrame, column: str) -> pd.DataFrame:
    def clean(text):
        text = standardize_text(text, ignore_chars=[","])
        if pd.isna(text):
            return "negativo"
        text = text.replace(";", ",")
        return resolve_contradictory_term(text, ",")

    df_copy = df.copy()
    df_copy[column] = impute(df_copy, [column], value="negativo")[column]
    df_copy[column] = df_copy[column].apply(clean)
    return encode_multilabel(df_copy, column, delimiter=",")


def clean_antecedentes_all(
    df: pd.DataFrame, columns: list = None
) -> pd.DataFrame:
    if columns is None:
        columns = ANTECEDENTES_COLS
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy = clean_antecedente(df_copy, col)
            df_copy = df_copy.drop(col, axis=1)
    return df_copy


# ── clean_antecedentes_anestesia ──────────────────────────────────────────────

def clean_antecedentes_anestesia(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = impute(df_copy, ["Antecedentes anestésicos"], value=0)
    df_copy.loc[
        df_copy["Antecedentes anestésicos"] == 0,
        ["Antecedentes quirúrgicos", "Anestesia previa"],
    ] = None
    # Drop free-text surgical history — it feeds clean_anestesia_previa upstream
    # and has no further use after that encoding step
    df_copy = df_copy.drop(columns=["Antecedentes quirúrgicos"], errors="ignore")
    return df_copy


# ── clean_anestesia_previa ────────────────────────────────────────────────────

ANESTESIA_WORD_MAP = {
    "anest": "anestesia", "anestesi": "anestesia", "anests": "anestesia",
    "anesteesia": "anestesia",
    "gral": "general", "grla": "general", "gralq": "general", "agral": "general",
    "genral": "general", "gnereal": "general", "agar": "general",
    "genereal": "general", "generalk": "general", "geberal": "general",
    "geeral": "general", "genberal": "general", "gneeral": "general",
    "genersal": "general", "gdneral": "general", "gfeneral": "general",
    "gfrla": "general", "agrla": "general", "genralk": "general",
    "gneral": "general", "gra": "general", "grl": "general", "geneal": "general",
    "enral": "general", "eneral": "general", "genera": "general",
    "generak": "general", "garl": "general", "gebneral": "general",
    "geeneral": "general", "geenral": "general", "gen": "general",
    "genearal": "general", "geneerl": "general", "genefral": "general",
    "genenral": "general", "gener": "general", "generael": "general",
    "generan": "general", "generanl": "general", "generasl": "general",
    "generl": "general", "generla": "general", "generral": "general",
    "genersl": "general", "genertal": "general", "genreal": "general",
    "genreral": "general", "genwral": "general", "gheneral": "general",
    "gnal": "general", "gre": "general", "greal": "general",
    "g": "general", "ag": "general", "a g": "general",
    "espinal ": "espinal", "esipinal": "espinal", "esipnal": "espinal",
    "espinl": "espinal", "espinsal": "espinal", "espnal": "espinal",
    "a wspinal": "espinal", "a espiunal": "espinal",
    "raquida": "raquidea", "raquidde": "raquidea", "raquide": "raquidea",
    "raqiodea": "raquidea", "raquiea": "raquidea", "raquieda": "raquidea",
    "raquiudea": "raquidea", "raqwuidea": "raquidea", "rquidea": "raquidea",
    "raquideaa": "raquidea", "raquideaq": "raquidea", "raquides": "raquidea",
    "rzaquidea": "raquidea", "aquidea": "raquidea", "ar": "raquidea",
    "epidrual": "epidural", "aepidural": "epidural",
    "peridrual": "peridural", "peridual": "peridural", "peidurtal": "peridural",
    "peridural ": "peridural",
    "neuaxial": "neuroaxial", "neuroaxiual": "neuroaxial", "neuroiaxial": "neuroaxial",
    "connductiva": "conductiva", "co nductiva": "conductiva",
    "conductivaraquidea": "conductiva raquidea",
    "conductivasedacion": "conductiva sedacion",
    "conductivosedacion": "conductiva sedacion",
    "conductivosedación": "conductiva sedacion",
    "conductivo": "conductiva",
    "alocal": "local", "loical": "local",
    "sed": "sedacion", "sedac": "sedacion", "sedacin": "sedacion",
    "sedacio": "sedacion", "sedacoin": "sedacion", "sedadcion": "sedacion",
    "sedacaion": "sedacion", "sedaicion": "sedacion", "sedcion": "sedacion",
    "sesacion": "sedacion", "sdeacion": "sedacion", "sdacion": "sedacion",
    "alsedacion": "sedacion", "sedacionlocal": "sedacion local",
    "localsedacion": "local sedacion", "localsedasdcion": "local sedacion",
    "loclasedacion": "local sedacion",
    "sedacionintraarticular": "sedacion intraarticular",
    "espinalsedacion": "espinal sedacion",
    "peridurlasedacion": "peridural sedacion",
    "raquideaepidural": "raquidea epidural",
    "agbloqueo": "general bloqueo",
    "epud": "epidural", "a": "", "nan": "desconocido",
}

ANESTESIA_KEY_WORDS = [
    "general", "raquidea", "epidural", "peridural", "regional", "local",
    "sedacion", "topica", "peribulbar", "intraarticular", "neuroaxial",
    "bloqueo", "bag", "conductiva", "espinal",
]


def clean_anestesia_previa(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["Anestesia previa"] = df_copy["Anestesia previa"].apply(standardize_text)
    df_copy["Anestesia previa"] = df_copy["Anestesia previa"].apply(
        lambda x: normalize_tokens(x, ANESTESIA_WORD_MAP)
    )
    df_copy["Anestesia previa"] = df_copy["Anestesia previa"].apply(
        lambda x: create_multi_label(x, ANESTESIA_KEY_WORDS, "desconocido")
    )
    df_copy = encode_multilabel(df_copy, "Anestesia previa", delimiter=",")
    df_copy = df_copy.drop(columns=["Anestesia previa"], errors="ignore")
    return df_copy


# ── clean_examenes ────────────────────────────────────────────────────────────

def clean_examenes(
    df: pd.DataFrame,
    examen_col: str = "Examen",
    resultado_col: str = "Resultado de examen de laboratorio",
) -> pd.DataFrame:
    df_copy = df.copy()

    def procesar_fila(row):
        examen = row[examen_col]
        resultado = row[resultado_col]
        if examen not in EXAM_PARSERS:
            return row
        cleaner_fn = EXAM_PARSERS[examen]
        val = cleaner_fn(resultado)
        return {**row, f"Examen_{examen}": val}

    df_resultado = df_copy.apply(procesar_fila, axis=1, result_type="expand")
    examen_cols = [c for c in df_resultado.columns if c.startswith("Examen_")]
    df_resultado[examen_cols] = df_resultado[examen_cols].fillna(0)
    df_resultado = df_resultado.drop(columns=[examen_col, resultado_col])
    return df_resultado


# ── clean_categorical_airway_assessment ───────────────────────────────────────

def clean_categorical_airway_assessment(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc["Cuello Móvil"] = dfc["Cuello Móvil"].fillna("Normal")
    dfc = encode_label_with_classes(dfc, "Cuello Móvil", classes=["Normal", "Reducido", "Fijo"])
    dfc = dfc.drop(columns=["Cuello Móvil"])
    dfc = encode_label_with_classes(dfc, "Apertura Oral", classes=["> 3 cm", "< 3 cm"])
    dfc = dfc.drop(columns=["Apertura Oral"])
    dfc["Puntaje Mallampati"] = dfc["Puntaje Mallampati"].apply(lambda x: x - 1)
    return dfc


# ── clean_categorical_physical_exam ──────────────────────────────────────────

def clean_categorical_physical_exam(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc["Estado Nutricional"] = dfc["Estado Nutricional"].apply(standardize_text)
    dfc["Estado Nutricional"] = dfc.apply(
        lambda row: "desnutrido" if row["IMC"] < 16
        else ("sobrepeso" if row["IMC"] > 30 else "normal"),
        axis=1,
    )
    dfc = encode_label_with_classes(
        dfc, "Estado Nutricional", classes=["desnutrido", "normal", "sobrepeso"]
    )
    dfc["Estado Nutricional_encoded"] = dfc["Estado Nutricional_encoded"].apply(
        lambda x: x - 1
    )
    dfc = dfc.drop(columns=["Estado Nutricional"])

    dfc["Color de Piel"] = dfc["Color de Piel"].apply(standardize_text)
    dfc = encode_onehot(dfc, "Color de Piel")
    dfc = dfc.drop(columns=["Color de Piel"])

    dfc["Sistema Nervioso"] = dfc["Sistema Nervioso"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Sistema Nervioso"] = dfc["Sistema Nervioso"].fillna("Sin dato")
    dfc = encode_multilabel(dfc, "Sistema Nervioso", delimiter=",")
    return dfc


# ── clean_categorical_consciousness ──────────────────────────────────────────

def clean_categorical_consciousness(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    dfc["Apertura Visual"] = dfc["Apertura Visual"].apply(standardize_text)
    dfc["Apertura Visual"] = dfc["Apertura Visual"].fillna("sin dato")
    dfc = encode_label_with_classes(
        dfc, "Apertura Visual",
        classes=["sin dato", "espontanea", "a la voz", "al dolor", "ninguna"],
    )
    dfc = dfc.drop(columns=["Apertura Visual"])

    dfc["Respuesta Verbal"] = dfc["Respuesta Verbal"].apply(standardize_text)
    dfc["Respuesta Verbal"] = dfc["Respuesta Verbal"].fillna("sin dato")
    dfc = encode_label_with_classes(
        dfc, "Respuesta Verbal",
        classes=[
            "sin dato", "conversacion orientada", "conversacion confusa",
            "palabras inapropiadas", "ruidos incomprensibles", "ninguna",
        ],
    )
    dfc = dfc.drop(columns=["Respuesta Verbal"])

    dfc["Respuesta Motora"] = dfc["Respuesta Motora"].apply(standardize_text)
    dfc["Respuesta Motora"] = dfc["Respuesta Motora"].fillna("sin dato")
    dfc = encode_label_with_classes(
        dfc, "Respuesta Motora",
        classes=[
            "sin dato", "obedece ordenes", "localiza el dolor",
            "retira al dolor", "flexiona al dolor", "flacido",
        ],
    )
    dfc = dfc.drop(columns=["Respuesta Motora"])
    return dfc


# ── clean_categorical_systems ─────────────────────────────────────────────────

def clean_categorical_systems(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    dfc["Sistema Respiratorio"] = dfc["Sistema Respiratorio"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Sistema Respiratorio"] = dfc["Sistema Respiratorio"].str.replace(";", "", regex=False)
    dfc["Sistema Respiratorio"] = dfc["Sistema Respiratorio"].fillna("normal")
    dfc["Disnea"] = dfc["Sistema Respiratorio"].apply(
        lambda x: "x" if "disnea" in str(x).lower() else x
    )
    dfc = encode_multilabel(dfc, "Sistema Respiratorio", delimiter=",")
    dfc = dfc.drop(columns=["Sistema Respiratorio"])

    dfc["Sistema cardiovascular"] = dfc["Sistema cardiovascular"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Sistema cardiovascular"] = dfc["Sistema cardiovascular"].str.replace(";", "", regex=False)
    dfc["Sistema cardiovascular"] = dfc["Sistema cardiovascular"].fillna("normal")
    dfc = encode_multilabel(dfc, "Sistema cardiovascular", delimiter=",")
    dfc = dfc.drop(columns=["Sistema cardiovascular"])

    dfc["Abdomen"] = dfc["Abdomen"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Abdomen"] = dfc["Abdomen"].fillna("normal")
    dfc = encode_multilabel(dfc, "Abdomen", delimiter=",")
    dfc = dfc.drop(columns=["Abdomen"])
    return dfc


# ── clean_categorical_specific_conditions ────────────────────────────────────

def clean_categorical_specific_conditions(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    dfc = encode_label_with_classes(dfc, "Arritmia", classes=["No", "Si"])
    dfc = dfc.drop(columns=["Arritmia"])

    dfc["Disnea"] = dfc["Disnea"].apply(standardize_text)
    dfc["Disnea"] = dfc["Disnea"].map({"x": "si"})
    dfc["Disnea"] = dfc["Disnea"].fillna("no")
    dfc = encode_label_with_classes(dfc, "Disnea", classes=["no", "si"])
    dfc = dfc.drop(columns=["Disnea"])

    dfc["Angina"] = dfc["Angina"].apply(standardize_text)
    dfc["Angina"] = dfc["Angina"].map({"x": "si"})
    dfc["Angina"] = dfc["Angina"].fillna("no")
    dfc = encode_label_with_classes(dfc, "Angina", classes=["no", "si"])
    dfc = dfc.drop(columns=["Angina"])

    dfc["Condición"] = dfc["Condición"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Condición"] = dfc["Condición"].str.replace(";", ",", regex=False)
    dfc["Condición"] = dfc["Condición"].fillna("ninguna")
    dfc = encode_multilabel(dfc, "Condición", delimiter=",")
    dfc = dfc.drop(columns=["Condición"])
    return dfc


# ── clean_categorical_procedures ─────────────────────────────────────────────

def clean_categorical_procedures(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()

    dfc["Tipo de anestesia propuesta"] = dfc["Tipo de anestesia propuesta"].apply(
        lambda x: standardize_text(x, remove_non_letters_flag=False)
    )
    dfc["Tipo de anestesia propuesta"] = dfc["Tipo de anestesia propuesta"].fillna("sin dato")
    dfc = encode_multilabel(dfc, "Tipo de anestesia propuesta", delimiter=",")
    dfc = dfc.drop(columns=["Tipo de anestesia propuesta"])

    dfc["Prótesis Dental"] = dfc["Prótesis Dental"].apply(standardize_text)
    dfc = encode_onehot(dfc, "Prótesis Dental")
    dfc = dfc.drop(columns=["Prótesis Dental"])
    return dfc


# ── clean_alergenos ───────────────────────────────────────────────────────────

def _remove_contradictions_alergeno(text: str):
    negatives = {
        "ninguna", "ninguno", "niguno", "ninugno", "ningueno", "ninguuno", "ningu",
        "ningúno", "ningúno.", "ningun", "ninguo", "niega", "niegan",
        "no", "no sabe", "no recuerda", "no refiere", "no refire", "no tiene",
        "no tiene alergia", "no tiene alergias", "no alergias", "no alergia",
        "negado", "negada", "negados", "negadas", "negativos", "negativas",
        "no reporta", "no refiere alergias", "no indica", "no consigna",
        "no menciona", "no se reporta", "no describe", "no señala",
        "no consigna alergias", "no refiere antecedente alérgico",
        "no refiere antecedentes alérgicos", "no refiere antecedentes de alergia",
        "nada", "no aplica", "sin dato", "sin datos", "sin alergias",
        "sin antecedentes de alergia", "sin antecedentes alérgicos", "desconocido",
        "ninguna.", "ninguno.", "niega alergias", "niega antecedentes alérgicos",
        "niega antecedentes de alergia", "ningun antecedente", "ningún antecedente",
        "ningun antecedente alergico", "ningún antecedente alérgico",
        "ningún antecedente de alergia", "-", "*", ".", "",
    }
    if text in negatives:
        return None
    return text


def _standardize_alergeno(s: str):
    if not s or isinstance(s, float):
        return None
    s = standardize_text(s, replace_char=" ")
    s = _remove_contradictions_alergeno(s)
    if not s or isinstance(s, float):
        return None
    from src.cleaning.text_utils import remove_stopwords_es
    s = remove_stopwords_es(
        s,
        extra_stopwords={
            "medio", "medios", "compuesta", "complejo", "oral", "b", "c",
            "recuerda", "alergica", "alergia", "medicamentos", "acido",
            "picadura", "plaquetas", "simple",
        },
    )
    return s


def _split_allergens(s: str) -> list:
    s = _standardize_alergeno(s)
    if not s:
        return []
    parts = re.split(r"[,/;\\\-]+|\s+Y\s+|\s+", s)
    return [p.strip() for p in parts if p.strip()]


ALLERGEN_ALIAS_MAP = {
    'penicilina': 'penicilina', 'penicilinas': 'penicilina', 'ampicilina': 'penicilina',
    'amoxicilina': 'penicilina', 'dicloxacilina': 'penicilina', 'cloxacilina': 'penicilina',
    'benzatinica': 'penicilina', 'benzatinica penicilina': 'penicilina',
    'pencilina': 'penicilina', 'peniclina': 'penicilina', 'penicillina': 'penicilina',
    'penicinlina': 'penicilina', 'penicilcina': 'penicilina', 'peniciclina': 'penicilina',
    'penicilinca': 'penicilina', 'amoxacilina': 'penicilina', 'amoxicilin': 'penicilina',
    'amoxicilna': 'penicilina', 'clavulanico': 'clavulanico',
    'ac clavulanico': 'clavulanico', 'amoxicilina clavulanico': 'amoxicilina_clavulanico',
    'cefadroxilo': 'cefalosporina', 'cefadroxil': 'cefalosporina',
    'cefalexina': 'cefalosporina', 'cefazolina': 'cefalosporina',
    'cefuroxima': 'cefalosporina', 'ceftriaxona': 'cefalosporina',
    'ceftazidima': 'cefalosporina', 'cefepime': 'cefalosporina',
    'cefalosporinas': 'cefalosporina', 'cefalosporina': 'cefalosporina',
    'vancomicina': 'glicopeptido', 'clindamicina': 'lincosamida',
    'metronidazol': 'nitroimidazol', 'tinidazol': 'nitroimidazol',
    'nitrofurantoina': 'nitrofurano', 'tetraciclina': 'tetraciclina',
    'azitromicina': 'macrolido', 'claritromicina': 'macrolido',
    'eritromicina': 'macrolido', 'ciprofloxacina': 'quinolona',
    'levofloxacina': 'quinolona', 'moxifloxacina': 'quinolona',
    'trimetoprim': 'trimetoprim', 'sulfametoxazol': 'sulfonamida',
    'smx': 'sulfonamida', 'tmp': 'trimetoprim', 'meropenem': 'carbapenem',
    'meropenen': 'carbapenem', 'nitazoxanida': 'antiparasitario',
    'albendazol': 'antiparasitario', 'fluconazol': 'azole', 'fluconzaol': 'azole',
    'ketoconazol': 'azole', 'ibuprofeno': 'aines', 'naproxeno': 'aines',
    'diclofenaco': 'aines', 'nimesulida': 'aines', 'celecoxib': 'aines',
    'etoricoxib': 'aines', 'eterocoxib': 'aines', 'eterocioxib': 'aines',
    'asa': 'aines', 'paracetamol': 'analgesico', 'acetaminofen': 'analgesico',
    'dolex': 'analgesico', 'sinalgen': 'analgesico', 'dipirona': 'analgesico',
    'dipyrona': 'analgesico', 'tramadol': 'opioide', 'trmadol': 'opioide',
    'meperidina': 'opioide', 'morfina': 'opioide', 'fentanilo': 'opioide',
    'hidromorfona': 'opioide', 'opiodes': 'opioide', 'opioides': 'opioide',
    'metoclopramida': 'antiemetico', 'plasil': 'antiemetico',
    'ondansetron': 'antiemetico', 'dimenhidrinato': 'antiemetico',
    'cetirizina': 'antihistaminico', 'loratadina': 'antihistaminico',
    'hidroxicina': 'antihistaminico', 'ranitidina': 'antiulceroso',
    'omeprazol': 'ppi', 'esomeprazol': 'ppi', 'amitriptilina': 'neuromodulador',
    'pregabalina': 'neuromodulador', 'lidocaina': 'anestesico_local',
    'bupivacaina': 'anestesico_local', 'ropivacaina': 'anestesico_local',
    'articaina': 'anestesico_local', 'propofol': 'anestesia_general',
    'midazolam': 'sedante', 'relajantes': 'relajante_muscular',
    'relajantes_musculares': 'relajante_muscular', 'metocarbamol': 'relajante_muscular',
    'epinefrina': 'vasoactivo', 'salbutamol': 'broncodilatador',
    'rituximab': 'biologico', 'gadobutrol': 'contraste_gadolinio',
    'contraste': 'contraste', 'yodo': 'yodo', 'iodo': 'yodo', 'yodado': 'yodo',
    'yodados': 'yodo', 'clorhexidina': 'antiseptico', 'alcohol': 'antiseptico',
    'tiomerzal': 'conservante', 'benzoato': 'conservante', 'micropore': 'adhesivo',
    'microporo': 'adhesivo', 'tegaderm': 'adhesivo', 'esparadrapo': 'adhesivo',
    'latex': 'latex', 'niquel': 'metal', 'huevo': 'alimento_huevo',
    'leche': 'alimento_lacteo', 'lactosa': 'alimento_lacteo',
    'soya': 'alimento_leguminosa', 'mani': 'frutos_secos_mani',
    'nuez': 'frutos_secos', 'almendra': 'frutos_secos', 'avellana': 'frutos_secos',
    'pistacho': 'frutos_secos', 'castana': 'frutos_secos',
    'mariscos': 'alimento_marisco', 'camaron': 'alimento_marisco',
    'camarones': 'alimento_marisco', 'pescado': 'alimento_pescado',
    'trigo': 'alimento_trigo', 'gluten': 'alimento_trigo',
    'fresa': 'alimento_fruta', 'fresas': 'alimento_fruta',
    'kiwi': 'alimento_fruta', 'chocolate': 'alimento_fruta',
    'cafeina': 'alimento_estimulante', 'polvo': 'ambiental', 'acaros': 'ambiental',
    'polen': 'ambiental', 'moho': 'ambiental', 'frio': 'ambiental',
    'abeja': 'animal', 'avispa': 'animal', 'perro': 'animal', 'gato': 'animal',
    'hierro': 'suplemento', 'calcio': 'suplemento', 'proteina': 'suplemento',
    'sulfa': 'sulfonamida', 'sulfas': 'sulfonamida', 'antibiotico': 'antibiotico',
    'aspirina': 'aines', 'piroxicam': 'aines', 'piroxican': 'aines',
    'meloxicam': 'aines', 'advil': 'aines', 'tramal': 'opioide',
    'codeina': 'opioide', 'hidrocodona': 'opioide', 'apronax': 'aines',
    'dipiron': 'analgesico', 'lisalgil': 'analgesico',
    'hioscina': 'antiespasmodico', 'buscapina': 'antiespasmodico',
    'trimebutina': 'antiespasmodico', 'dexametasona': 'corticosteroide',
    'hidrocortisona': 'corticosteroide', 'ketotifeno': 'antihistaminico',
    'haloperidol': 'antipsicotico', 'nefopam': 'analgesico_no_opioide',
    'tizanidina': 'relajante_muscular', 'lamotrigina': 'antiepileptico',
    'carbamazepina': 'antiepileptico', 'tiamina': 'suplemento',
    'vitamina': 'suplemento', 'terbutalina': 'broncodilatador',
    'ketamina': 'anestesia_general', 'trimetropin': 'trimetoprim',
    'doxiciclina': 'tetraciclina', 'cipro': 'quinolona',
    'gentamicina': 'aminoglucosido', 'amikacina': 'aminoglucosido',
    'unasyn': 'penicilina', 'bencetazil': 'penicilina', 'clavulin': 'penicilina',
    'isodine': 'yodo', 'mertiolate': 'conservante', 'espadadrapo': 'adhesivo',
    'manillas': 'adhesivo', 'pina': 'alimento_fruta', 'lacteos': 'alimento_lacteo',
    'vaca': 'alimento_lacteo', 'cerdo': 'alimento_carne', 'pollo': 'alimento_carne',
    'nueces': 'frutos_secos', 'insectos': 'animal', 'abejas': 'animal',
    'rinitis': 'ambiental', 'antigripales': 'descongestionante',
    'aines': 'aines', 'quinolonas': 'quinolona', 'macrodantina': 'nitrofurano',
    'sulbactam': 'inhibidor_betalactamasa', 'asparaginasa': 'quimioterapia',
    'prednisolona': 'corticosteroide', 'dristan': 'descongestionante',
    'pelo': 'ambiental', 'semillas': 'alimento_semillas',
    'semillas girasol': 'alimento_semillas', 'aji': 'alimento_especia',
    'canela': 'alimento_especia', 'menta': 'alimento_especia',
    'pimenton': 'alimento_especia', 'jengibre': 'alimento_especia',
    'limoncillo': 'alimento_especia', 'durazno': 'alimento_fruta',
    'mora': 'alimento_fruta', 'guayaba': 'alimento_fruta',
    'queso cheddar': 'alimento_lacteo', 'embutidos': 'alimento_carne',
    'pan': 'alimento_trigo', 'pepino': 'alimento_vegetal',
    'champinones': 'alimento_vegetal', 'langosta': 'alimento_marisco',
    'lagostinos': 'alimento_marisco', 'crustaceos': 'alimento_marisco',
    'atun': 'alimento_pescado', 'caseina': 'alimento_lacteo',
    'alimentos': 'alimento_general', 'alimentarias': 'alimento_general',
    'ambientales': 'ambiental', 'ambiente': 'ambiental',
    'medioambientales': 'ambiental', 'aire acondicionado': 'ambiental',
    'esporas pasto': 'ambiental', 'heno': 'ambiental',
    'matas olores fuertes': 'ambiental', 'hormiga': 'animal',
    'hormigas': 'animal', 'conejos': 'animal', 'picadas': 'animal',
    'desodorante': 'producto_topico', 'lociones': 'producto_topico',
    'lociones cosmeticas': 'producto_topico',
    'peroxido benzoilo': 'producto_topico', 'minoxidil': 'producto_topico',
    'tinte': 'producto_topico', 'tinte cabello': 'producto_topico',
    'curitas': 'adhesivo', 'bandas elasticas': 'adhesivo',
    'micopororo': 'adhesivo', 'vicryl': 'material_medico',
    'vycril': 'material_medico', 'vicryl sutura': 'material_medico',
    'electrodos papel': 'material_medico', 'ponstan': 'aines',
    'postan': 'aines', 'profenid': 'aines', 'ketoprofeno': 'aines',
    'ketorolaco': 'aines', 'etofenamato': 'aines', 'indometacina': 'aines',
    'feldene': 'aines', 'oxaprozina': 'aines', 'acetil salicilico': 'aines',
    'acetilsalisilico': 'aines', 'analgesicos': 'analgesico',
    'analgesico': 'analgesico', 'narcoticos': 'opioide', 'codeine': 'opioide',
    'coidena': 'opioide', 'paracodina': 'opioide', 'oxicodona': 'opioide',
    'morifna': 'opioide', 'novalgina': 'analgesico', 'neozaldina': 'analgesico',
    'dololed': 'analgesico', 'losalgil': 'analgesico',
    'antihistaminicos': 'antihistaminico', 'allegra': 'antihistaminico',
    'claritine': 'antihistaminico', 'difenhidramina': 'antihistaminico',
    'tavegyl': 'antihistaminico', 'tabegil': 'antihistaminico',
    'mareol': 'antihistaminico', 'listerine': 'antiseptico',
    'dimetapp': 'descongestionante', 'noxpirin': 'descongestionante',
    'noxpirina': 'descongestionante', 'noraver': 'descongestionante',
    'noraver masticable': 'descongestionante', 'afrin': 'descongestionante',
    'pesudoefedrina': 'descongestionante', 'phenilifrina': 'descongestionante',
    'fenilefrina': 'descongestionante', 'ambroxol': 'mucolitico',
    'ambroxol pediatrico': 'mucolitico', 'fluimucil': 'mucolitico',
    'nacetilcisteina': 'mucolitico', 'levodropropicina': 'antitusivo',
    'keflex': 'cefalosporina', 'kefelx': 'cefalosporina',
    'rocefin': 'cefalosporina', 'bactrim': 'sulfonamida',
    'bactrym': 'sulfonamida', 'bactirin': 'sulfonamida',
    'bactirim': 'sulfonamida', 'benzetacil': 'penicilina',
    'benzetacyl': 'penicilina', 'tazocin': 'penicilina',
    'fosfomicina': 'antibiotico', 'cloranfenicol': 'antibiotico',
    'polimixina': 'antibiotico', 'estreptomicina': 'aminoglucosido',
    'zitromax': 'macrolido', 'rulid': 'macrolido', 'terramicina': 'tetraciclina',
    'terramicina oftalmica': 'tetraciclina', 'nalinixico': 'quinolona',
    'nailinixico': 'quinolona', 'tms': 'sulfonamida',
    'betalactamicos': 'antibiotico_betalactamico', 'atb': 'antibiotico',
    'nitaxuzanida': 'antiparasitario', 'nitaxazanida': 'antiparasitario',
    'nitazoxadina': 'antiparasitario', 'notazoxamida': 'antiparasitario',
    'secnidal': 'antiparasitario', 'ivermectina': 'antiparasitario',
    'pirantel': 'antiparasitario', 'aciclovir': 'antiviral',
    'betametasona': 'corticosteroide', 'budesonida': 'corticosteroide',
    'beclometasona inhalada': 'corticosteroide', 'esteroides': 'corticosteroide',
    'corticoesteroides': 'corticosteroide', 'pentotal': 'anestesia_general',
    'sevoflorane': 'anestesia_general', 'xilocaina': 'anestesico_local',
    'succinilcolina': 'relajante_muscular', 'protamina': 'hematologico',
    'tranexamico': 'hemostatico', 'dobutamina': 'vasoactivo',
    'clenbuterol': 'broncodilatador', 'terbutamila': 'broncodilatador',
    'montelukast': 'antileucotrieno', 'duloxetina': 'antidepresivo',
    'sertralina': 'antidepresivo', 'zopiclona': 'hipnotico',
    'topiramato': 'antiepileptico', 'fenitoina': 'antiepileptico',
    'epamin': 'antiepileptico', 'valproico': 'antiepileptico',
    'atorvastatina': 'hipolipemiante', 'rosuvastatina': 'hipolipemiante',
    'estatinas': 'hipolipemiante', 'gemfibrozilo': 'hipolipemiante',
    'enalapril': 'cardiovascular', 'captopril': 'cardiovascular',
    'verapamilo': 'cardiovascular', 'nifedipino amlodipino': 'cardiovascular',
    'iecas': 'cardiovascular', 'furosemida': 'diuretico',
    'hidroclorotiazida': 'diuretico', 'hidroclorotiacida': 'diuretico',
    'hctz': 'diuretico', 'espironolactona': 'diuretico',
    'tamsulosina': 'urologico', 'mirabregon': 'urologico',
    'heparina': 'anticoagulante', 'sucralfato': 'antiulceroso',
    'gaviscon': 'antiulceroso', 'gaviston': 'antiulceroso',
    'salicilato bismuto': 'antiulceroso', 'alka seltzer': 'antiulceroso',
    'lomotil': 'antidiarreico', 'loperamida': 'antidiarreico',
    'carboplatino': 'quimioterapia', 'oxaliplatino': 'quimioterapia',
    'sorafenib': 'quimioterapia', 'metotrexate': 'quimioterapia',
    'lenalinomida': 'quimioterapia', 'ertapenem': 'carbapenem',
    'ertapenen': 'carbapenem', 'pertuzumab': 'biologico',
    'toxoide tetanico': 'vacuna', 'vacuna pentavalente': 'vacuna',
    'sangre': 'biologico', 'plasma': 'biologico', 'omega': 'suplemento',
    'cianocobalamina': 'suplemento', 'ascorbico': 'suplemento',
    'sulfato ferroso': 'suplemento', 'propoleo': 'alimento_otros',
    'miel': 'alimento_otros', 'antiparasitario': 'antiparasitario',
    'antidepresivos': 'antidepresivo',
    'antihistaminicos mareol': 'antihistaminico',
    'cromus': 'otro', 'wintomilom': 'otro', 'claridad': 'otro',
    'purgante': 'otro', 'analgesicos esteroideos': 'corticosteroide',
    'reaccion cutanea': 'otro', 'dermatitis': 'otro',
    'dermatitis contacto': 'otro', 'atopica': 'otro', 'atopico': 'otro',
    'atopia': 'otro', 'desconoce': 'otro', 'med': 'otro',
    'mencionados': 'otro', 'nombre': 'otro', 'varios': 'otro',
    'concentrado': 'otro', 'ver ea': 'otro', 'ver anexo': 'otro',
    'refiere sustancia': 'otro', 'sabe medicamento': 'otro',
    'estudio alergias': 'otro', 'medicamento malaria': 'antiparasitario',
}

ALLERGEN_REGEX_RULES = [
    (re.compile(r".*cilin.*"),                                        "penicilina"),
    (re.compile(r"^cef.*"),                                           "cefalosporina"),
    (re.compile(r".*floxac.*"),                                       "quinolona"),
    (re.compile(r".*clavulan.*"),                                     "clavulanico"),
    (re.compile(r".*azol.*"),                                         "azole"),
    (re.compile(r".*metronidaz.*"),                                   "nitroimidazol"),
    (re.compile(r".*nitrofur.*"),                                     "nitrofurano"),
    (re.compile(r".*tetra.*ciclin.*"),                                "tetraciclina"),
    (re.compile(r".*vancomicin.*"),                                   "glicopeptido"),
    (re.compile(r".*clindamicin.*"),                                  "lincosamida"),
    (re.compile(r".*cetirizin.*|.*loratadin.*|.*hidroxizin.*"),       "antihistaminico"),
    (re.compile(r".*ibuprof.*|.*naprox.*|.*diclofenac.*|.*coxib.*|.*nimesul.*"), "aines"),
    (re.compile(r".*tramad.*|.*morf.*|.*fentan.*|.*meperid.*|.*hidromorf.*"),    "opioide"),
    (re.compile(r".*lidocain.*|.*bupivacain.*|.*ropivacain.*|.*articain.*"),     "anestesico_local"),
    (re.compile(r".*yod.*|.*iod.*"),                                  "yodo"),
    (re.compile(r".*gad.*"),                                          "contraste_gadolinio"),
    (re.compile(r".*clorhexid.*"),                                    "antiseptico"),
    (re.compile(r".*micropor.*|.*esparadrap.*|.*tegaderm.*"),         "adhesivo"),
    (re.compile(r".*lat[eé]x.*"),                                     "latex"),
    (re.compile(r".*niquel.*"),                                       "metal"),
    (re.compile(r".*acaro.*|.*polvo.*|.*polen.*|.*moho.*"),           "ambiental"),
    (re.compile(r".*camaron.*|.*marisc.*"),                           "alimento_marisco"),
    (re.compile(r"^pnc$"),                                            "penicilina"),
    (re.compile(r"^antiinflamatorios?$"),                             "aines"),
    (re.compile(r"^tms$"),                                            "sulfonamida"),
    (re.compile(r"^iecas?$"),                                         "cardiovascular"),
    (re.compile(r"^hctz$"),                                           "diuretico"),
]

ALLERGEN_COLLAPSE_MAP = {
    "alimento_huevo": "alimentos", "alimento_lacteo": "alimentos",
    "alimento_leguminosa": "alimentos", "frutos_secos": "alimentos",
    "frutos_secos_mani": "alimentos", "alimento_marisco": "alimentos",
    "alimento_pescado": "alimentos", "alimento_trigo": "alimentos",
    "alimento_fruta": "alimentos", "alimento_estimulante": "alimentos",
    "alimento_vegetal": "alimentos", "alimento_semillas": "alimentos",
    "alimento_otros": "alimentos", "alimento_general": "alimentos",
    "suplemento": "suplementos",
    "ambiental": "ambiental_animales", "animal": "ambiental_animales",
    "yodo": "contacto_hospitalario", "contraste": "contacto_hospitalario",
    "contraste_gadolinio": "contacto_hospitalario",
    "adhesivo": "contacto_hospitalario",
    "material_medico": "contacto_hospitalario", "latex": "contacto_hospitalario",
    "metal": "contacto_hospitalario", "producto_topico": "contacto_hospitalario",
    "antiseptico": "contacto_hospitalario", "conservante": "contacto_hospitalario",
    "colorante_diagnostico": "contacto_hospitalario",
    "antibiotico": "med_antiinfecciosos",
    "antibiotico_betalactamico": "med_antiinfecciosos",
    "penicilina": "med_antiinfecciosos", "cefalosporina": "med_antiinfecciosos",
    "carbapenem": "med_antiinfecciosos", "macrolido": "med_antiinfecciosos",
    "quinolona": "med_antiinfecciosos", "lincosamida": "med_antiinfecciosos",
    "glicopeptido": "med_antiinfecciosos", "nitroimidazol": "med_antiinfecciosos",
    "nitrofurano": "med_antiinfecciosos", "tetraciclina": "med_antiinfecciosos",
    "aminoglucosido": "med_antiinfecciosos", "azole": "med_antiinfecciosos",
    "sulfonamida": "med_antiinfecciosos", "antiparasitario": "med_antiinfecciosos",
    "antiviral": "med_antiinfecciosos",
    "aines": "med_analgesia_no_opioide", "analgesico": "med_analgesia_no_opioide",
    "analgesico_no_opioide": "med_analgesia_no_opioide",
    "antiespasmodico": "med_analgesia_no_opioide",
    "antitusivo": "med_analgesia_no_opioide",
    "opioide": "med_opioides",
    "anestesia_general": "med_anestesia_y_relajantes",
    "anestesico_local": "med_anestesia_y_relajantes",
    "sedante": "med_anestesia_y_relajantes",
    "relajante_muscular": "med_anestesia_y_relajantes",
    "antihistaminico": "med_alergias_respiratorio",
    "broncodilatador": "med_alergias_respiratorio",
    "antileucotrieno": "med_alergias_respiratorio",
    "descongestionante": "med_alergias_respiratorio",
    "mucolitico": "med_alergias_respiratorio",
    "antiemetico": "med_gastrointestinal", "antiulceroso": "med_gastrointestinal",
    "ppi": "med_gastrointestinal", "antidiarreico": "med_gastrointestinal",
    "probiotico": "med_gastrointestinal", "sucralfato": "med_gastrointestinal",
    "cardiovascular": "med_cardio_renal_hemostasia",
    "diuretico": "med_cardio_renal_hemostasia",
    "anticoagulante": "med_cardio_renal_hemostasia",
    "hemostatico": "med_cardio_renal_hemostasia",
    "hematologico": "med_cardio_renal_hemostasia",
    "hipolipemiante": "med_cardio_renal_hemostasia",
    "vasoactivo": "med_cardio_renal_hemostasia",
    "antiepileptico": "med_neuro_psiquiatria",
    "neuromodulador": "med_neuro_psiquiatria",
    "antidepresivo": "med_neuro_psiquiatria", "hipnotico": "med_neuro_psiquiatria",
    "urologico": "med_urologico",
    "quimioterapia": "onco_inmuno_vacunas", "biologico": "onco_inmuno_vacunas",
    "vacuna": "onco_inmuno_vacunas",
    "otro": "otro",
}


def _map_allergen_token(tok: str):
    if not tok:
        return None
    if tok in ALLERGEN_ALIAS_MAP:
        return ALLERGEN_ALIAS_MAP[tok]
    for rx, target in ALLERGEN_REGEX_RULES:
        if rx.fullmatch(tok) or rx.search(tok):
            return target
    return None


def _classify_allergen_text(text: str) -> list:
    toks = _split_allergens(text)
    classes = []
    has_other = False
    for t in toks:
        cls = _map_allergen_token(t)
        if cls:
            classes.append(cls)
        else:
            has_other = True
    if has_other:
        classes.append("otro")
    return sorted(set(c for c in classes if c))


def _collapse_classes(fine_classes: list, collapse_map: dict = None) -> str:
    if collapse_map is None:
        collapse_map = ALLERGEN_COLLAPSE_MAP
    macros = [collapse_map.get(c, "otro") for c in (fine_classes or [])]
    return ",".join(set(macros))


def clean_alergenos(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["Alérgeno"] = df_clean["Alérgeno"].apply(_classify_allergen_text)
    df_clean["Alérgeno"] = df_clean["Alérgeno"].apply(
        lambda x: _collapse_classes(x, ALLERGEN_COLLAPSE_MAP)
    )
    df_clean = encode_multilabel(df_clean, "Alérgeno")
    df_clean.drop(columns=["Alérgeno"], inplace=True)
    df_clean.drop(columns=["Alergias"], inplace=True, errors="ignore")
    return df_clean


# ── Pipeline completo ─────────────────────────────────────────────────────────

def clean_preop(df: pd.DataFrame, cleaning_cfg: dict) -> pd.DataFrame:
    """
    Pipeline completo de limpieza determinista del dataset preoperatorio.
    Replica fielmente el pipeline del notebook 2_data_cleaning.ipynb (cell 36).

    Las etapas que requieren APIs externas (clean_medicamentos, clean_dx_proc)
    se aplican desde src/cleaning/enrichment.py.
    """
    df = df.copy()

    # 0. Correcciones de encoding en nombres de columnas
    encoding_fixes = cleaning_cfg.get("encoding_fixes", {})
    if encoding_fixes:
        df = df.rename(columns=encoding_fixes)
        logger.info(f"Encoding fixes aplicados: {len(encoding_fixes)} columnas")

    # 1. Eliminar columnas que tienen una única categoría / leakage
    leakage_cols = [c for c in cleaning_cfg.get("leakage_columns", []) if c in df.columns]
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
        logger.info(f"Columnas de fuga eliminadas: {leakage_cols}")

    # 2. Limpiar fecha y hora de valoración
    if "Fecha" in df.columns and "Hora" in df.columns:
        df = clean_fecha_hora(df)
        logger.info("clean_fecha_hora OK")

    # 3. Codificar sexo y tipo de atención
    if "Sexo" in df.columns and "Atención" in df.columns:
        df = clean_sexo_atencion(df)
        logger.info("clean_sexo_atencion OK")

    # 3b. Aplicar outlier_rules del config: marcar valores fuera de rango como NaN
    outlier_rules = cleaning_cfg.get("outlier_rules", {})
    for col, bounds in outlier_rules.items():
        if col in df.columns:
            lo = bounds.get("min")
            hi = bounds.get("max")
            mask = pd.Series(False, index=df.index)
            if lo is not None:
                mask |= df[col] < lo
            if hi is not None:
                mask |= df[col] > hi
            df.loc[mask, col] = float("nan")
    if outlier_rules:
        logger.info(f"outlier_rules aplicadas: {list(outlier_rules.keys())}")

    # 4. Limpiar variables fisiológicas (edad, peso, talla, IMC, signos vitales)
    df = clean_variables_fisiologicas(df)
    logger.info("clean_variables_fisiologicas OK")

    # 5. Filtro de edad mínima
    age_filter = cleaning_cfg.get("age_filter", {})
    age_col = age_filter.get("column", "Edad")
    age_min = age_filter.get("min")
    if age_min is not None and age_col in df.columns:
        before = len(df)
        df = df[~(df[age_col] < age_min)].reset_index(drop=True)
        logger.info(f"Filtro edad ≥{age_min}: {before} → {len(df)} filas")

    # 6. Limpiar grupo sanguíneo, RH y NYHA
    if "Grupo Sanguíneo" in df.columns:
        df = clean_grupo_rh_y_nyha(df)
        logger.info("clean_grupo_rh_y_nyha OK")

    # 7. Limpiar antecedentes clínicos
    df = clean_antecedentes_all(df)
    logger.info("clean_antecedentes_all OK")

    # 8. Limpiar antecedentes anestésicos
    if "Antecedentes anestésicos" in df.columns:
        df = clean_antecedentes_anestesia(df)
        logger.info("clean_antecedentes_anestesia OK")

    # 9. Limpiar anestesia previa
    if "Anestesia previa" in df.columns:
        df = clean_anestesia_previa(df)
        logger.info("clean_anestesia_previa OK")

    # 10. Limpiar exámenes de laboratorio
    if "Examen" in df.columns:
        df = clean_examenes(df)
        logger.info("clean_examenes OK")

    # 11. Limpiar vía aérea
    if "Cuello Móvil" in df.columns:
        df = clean_categorical_airway_assessment(df)
        logger.info("clean_categorical_airway_assessment OK")

    # 12. Limpiar examen físico general
    if "Estado Nutricional" in df.columns:
        df = clean_categorical_physical_exam(df)
        logger.info("clean_categorical_physical_exam OK")

    # 13. Limpiar nivel de conciencia
    if "Apertura Visual" in df.columns:
        df = clean_categorical_consciousness(df)
        logger.info("clean_categorical_consciousness OK")

    # 14. Limpiar sistemas orgánicos
    if "Sistema Respiratorio" in df.columns:
        df = clean_categorical_systems(df)
        logger.info("clean_categorical_systems OK")

    # 15. Limpiar condiciones específicas
    if "Arritmia" in df.columns:
        df = clean_categorical_specific_conditions(df)
        logger.info("clean_categorical_specific_conditions OK")

    # 16. Limpiar procedimientos propuestos
    if "Tipo de anestesia propuesta" in df.columns:
        df = clean_categorical_procedures(df)
        logger.info("clean_categorical_procedures OK")

    # 17. Limpiar alérgenos
    if "Alérgeno" in df.columns:
        df = clean_alergenos(df)
        logger.info("clean_alergenos OK")

    # 18. Eliminar columnas identificadoras al final (preserva joins intermedios)
    id_cols = [c for c in cleaning_cfg.get("identifier_columns", []) if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        logger.info(f"Columnas identificadoras eliminadas: {id_cols}")

    logger.info(f"Pipeline limpieza completo: {df.shape}")
    return df
