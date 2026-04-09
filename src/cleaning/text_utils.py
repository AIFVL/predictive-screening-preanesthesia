# src/cleaning/text_utils.py
"""
Utilidades de limpieza y normalización de texto.
Migradas fielmente desde notebooks/2_data_cleaning.ipynb (cells 9, 30).
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


# ── Normalización básica ──────────────────────────────────────────────────────

def remove_accents(text: str) -> str:
    if pd.isna(text):
        return text
    nfkd_form = unicodedata.normalize("NFKD", str(text))
    return nfkd_form.encode("ASCII", "ignore").decode("ASCII")


def to_lower(text: str) -> str:
    if pd.isna(text):
        return text
    return str(text).lower()


def remove_non_letters(text: str, ignore_chars: list = None, replace_char: str = "") -> str:
    if pd.isna(text):
        return text
    if ignore_chars is None:
        ignore_chars = []
    ignore_pattern = "".join(re.escape(char) for char in ignore_chars)
    pattern = rf"[^a-zA-Z{ignore_pattern}\s]"
    return re.sub(pattern, replace_char, str(text))


def clean_whitespace(text: str) -> str:
    if pd.isna(text):
        return text
    return " ".join(str(text).split())


def standardize_text(
    text: str,
    remove_accents_flag: bool = True,
    to_lower_flag: bool = True,
    clean_whitespace_flag: bool = True,
    remove_non_letters_flag: bool = True,
    ignore_chars: list = None,
    replace_char: str = "",
) -> str:
    if pd.isna(text):
        return text
    result = str(text)
    if remove_accents_flag:
        result = remove_accents(result)
    if to_lower_flag:
        result = to_lower(result)
    if clean_whitespace_flag:
        result = clean_whitespace(result)
    if remove_non_letters_flag:
        result = remove_non_letters(result, ignore_chars or [], replace_char)
    return result


# ── Stopwords español ─────────────────────────────────────────────────────────

def remove_stopwords_es(text: str, extra_stopwords: set = None) -> str:
    if not isinstance(text, str):
        return ""
    import nltk
    from nltk.corpus import stopwords
    tokens = re.findall(r"\b\w+\b", text.lower())
    stop_words = set(stopwords.words("spanish"))
    if extra_stopwords:
        stop_words.update([w.lower() for w in extra_stopwords])
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


# ── Token helpers ─────────────────────────────────────────────────────────────

def normalize_tokens(text: str, word_map: dict) -> str:
    s = str(text).lower().strip()
    for wrong, right in word_map.items():
        s = re.sub(rf"\b{re.escape(wrong)}\b", right, s)
    return s


def create_multi_label(text: str, labels: list, default: str = "otros") -> str:
    if pd.isna(text):
        return default
    result = [label for label in labels if label in text]
    if not result:
        return default
    return ", ".join(result)


def resolve_contradictory_term(
    text: str, delimiter: str, contradictory_term: str = "negativo"
) -> str:
    items = [i.strip() for i in text.split(delimiter) if i.strip()]
    items = list(set(items))
    if (contradictory_term in items) and len(items) > 1:
        items = [i for i in items if i != contradictory_term]
    return delimiter.join(items)
