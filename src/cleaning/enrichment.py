# src/cleaning/enrichment.py
"""
Etapas de enriquecimiento del dataset preoperatorio que requieren APIs externas.
Migrado fielmente desde notebooks/2_data_cleaning.ipynb (cells 32, 34).

Funciones:
  - clean_medicamentos  → RxNav/RxNorm API + ATC hierarchy (cell 32)
  - clean_dx_proc       → GoogleTranslate + ClinicalTables NLM + BART-MNLI (cell 34)

Cada función usa CSV caches para evitar repetir llamadas a APIs.
Los paths de los caches son configurables (parámetro cache_dir).
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import csv
import time
from pathlib import Path

import pandas as pd

from src.cleaning.encoding_utils import encode_label_with_classes, encode_onehot
from src.cleaning.text_utils import remove_stopwords_es, standardize_text
from src.utils.logger import get_logger

logger = get_logger("cleaning.enrichment")

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
CT_BASE = "https://clinicaltables.nlm.nih.gov/api"
TIMEOUT = 12
SLEEP_BETWEEN = 0.10
HEADERS_JSON = {"Accept": "application/json"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades compartidas
# ─────────────────────────────────────────────────────────────────────────────

def _safe(s):
    return (
        None
        if (s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() == "")
        else str(s).strip()
    )


def _load_csv(path: str, columns: list) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


def _flush_append_csv(path: str, df_new: pd.DataFrame, subset_cols: list = None):
    if df_new is None or df_new.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        try:
            base = pd.read_csv(path, quoting=csv.QUOTE_ALL, on_bad_lines="skip")
        except Exception:
            base = pd.DataFrame(columns=df_new.columns)
        out = pd.concat([base, df_new], ignore_index=True)
    else:
        out = df_new.copy()
    if subset_cols:
        out = out.drop_duplicates(subset=subset_cols, keep="last")
    out.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# clean_medicamentos  (RxNav → ATC hierarchy one-hots)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text_for_lookup(s: str) -> str:
    s = s.lower().strip()
    return re.sub(r"\s+", " ", s)


def _rxnav_approximate_term(q: str):
    import requests
    q = _normalize_text_for_lookup(q)
    try:
        url = f"{RXNAV_BASE}/approximateTerm.json"
        resp = requests.get(url, params={"term": q, "maxEntries": 1, "option": 1}, timeout=TIMEOUT)
        data = resp.json()
        cand = (data.get("approximateGroup") or {}).get("candidate") or []
        if not cand:
            return None
        rxcui = cand[0].get("rxcui")
        if not rxcui:
            return None
        prop = requests.get(
            f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json", timeout=TIMEOUT
        ).json()
        p = prop.get("properties") or {}
        return {
            "rxcui": p.get("rxcui") or rxcui,
            "rxnorm_name": p.get("name"),
            "rxnorm_tty": p.get("tty"),
        }
    except Exception:
        return None


def _rxclass_by_rxcui_atc(rxcui: str):
    import requests
    try:
        url = f"{RXNAV_BASE}/rxclass/class/byRxcui.json"
        resp = requests.get(url, params={"rxcui": rxcui}, timeout=TIMEOUT)
        data = resp.json()
        out = []
        for item in (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", []) or []:
            mc = item.get("rxclassMinConceptItem") or {}
            ctype = mc.get("classType")
            if ctype and ctype.startswith("ATC"):
                out.append({"classId": mc.get("classId"), "className": mc.get("className"),
                            "classType": ctype})
        return out
    except Exception:
        return []


def _atc_levels_from_code(atc_code: str):
    if not atc_code:
        return (None, None, None, None)
    s = atc_code
    l1 = s[0]
    l2 = s[:3] if len(s) >= 3 else None
    l3 = s[:4] if len(s) >= 4 else None
    l4 = s[:5] if len(s) >= 5 else None
    return (l1, l2, l3, l4)


def _pick_best_atc_hierarchy(atc_list):
    if not atc_list:
        return {}
    atc_sorted = sorted(
        [x for x in atc_list if x.get("classId")],
        key=lambda x: len(x["classId"]),
        reverse=True,
    )
    top = atc_sorted[0]
    l1, l2, l3, l4 = _atc_levels_from_code(top["classId"])
    return {
        "atc_l1_code": l1, "atc_l1_name": None,
        "atc_l2_code": l2, "atc_l2_name": None,
        "atc_l3_code": l3, "atc_l3_name": None,
        "atc_l4_code": l4, "atc_l4_name": None,
    }


def _fill_atc_names(hier, atc_list):
    code2name = {x["classId"]: x.get("className") for x in atc_list}
    for lvl in ("atc_l4", "atc_l3", "atc_l2", "atc_l1"):
        c = hier.get(f"{lvl}_code")
        if c and c in code2name:
            hier[f"{lvl}_name"] = code2name[c]
    return hier


def _onehots_atc_l1(letter):
    groups = list("ABCDGHJLMNRPSV")
    return {f"atc_l1_is_{g}": (1 if letter == g else 0) for g in groups}


def _initial_medicamentos_cleaning(text):
    def remove_contradictions_meds(t):
        negatives = {
            "ninguno", "ninguna", "niega", "niegan", "no refiere", "no refire",
            "no", "negado", "negados", "negativos", "nada", "no meds",
            "no medicamentos", "no toma", "no toma medicamentos", "no medicación",
            "no medicación.", "no toma medicación.", "ninguno.", "ninguna.",
            "no usa", "no recuerda", "no toma medicacion", "no t meds",
            "ninguno actualmente", "ninguno actualmente.",
            "no toma medicamentos.", "no medicamentos.", "-", "*", ".", "",
        }
        return None if t in negatives else t

    text = standardize_text(text, replace_char=" ")
    text = remove_contradictions_meds(text)
    if text is None:
        return None
    return remove_stopwords_es(text)


def rxnorm_features_for_med_list(
    med_list: list,
    cache_csv: str,
    sleep_sec: float = 0.15,
    use_cache: bool = True,
    cache_flush_every: int = 15,
) -> pd.DataFrame:
    """
    Resuelve cada medicamento con RxNorm → ATC hierarchy → one-hots L1.
    Usa un CSV cache para no repetir llamadas.
    """
    cols = [
        "input_raw", "rxnorm_rxcui", "rxnorm_name", "rxnorm_tty",
        "atc_l1_code", "atc_l1_name", "atc_l2_code", "atc_l2_name",
        "atc_l3_code", "atc_l3_name", "atc_l4_code", "atc_l4_name",
    ]
    cache = _load_csv(cache_csv, cols) if use_cache else pd.DataFrame(columns=cols)
    cache_idx = (
        {r.lower(): i for i, r in enumerate(cache["input_raw"].astype(str))}
        if not cache.empty else {}
    )

    rows = []
    new_cache_rows = []
    total = len(med_list)

    for count, raw in enumerate(med_list, 1):
        logger.debug(f"RxNorm {count}/{total}: {raw}")
        raw_clean = _safe(raw)
        if raw_clean is None:
            null_row = {"input_raw": raw, **{k: None for k in cols[1:]},
                        **_onehots_atc_l1(None)}
            rows.append(null_row)
            new_cache_rows.append({k: null_row[k] for k in cols})
            if use_cache and len(new_cache_rows) >= cache_flush_every:
                _flush_append_csv(cache_csv, pd.DataFrame(new_cache_rows))
                cache = _load_csv(cache_csv, cols)
                cache_idx = {r.lower(): i for i, r in enumerate(cache["input_raw"].astype(str))}
                new_cache_rows = []
            continue

        if use_cache and raw_clean.lower() in cache_idx:
            row = cache.iloc[cache_idx[raw_clean.lower()]].to_dict()
            rows.append(row | _onehots_atc_l1(row.get("atc_l1_code")))
            continue

        base = _rxnav_approximate_term(raw_clean)
        if not base:
            null_row = {"input_raw": raw_clean, **{k: None for k in cols[1:]},
                        **_onehots_atc_l1(None)}
            rows.append(null_row)
            new_cache_rows.append({k: null_row[k] for k in cols})
        else:
            atc_list = _rxclass_by_rxcui_atc(base["rxcui"])
            hier = _pick_best_atc_hierarchy(atc_list)
            hier = _fill_atc_names(hier, atc_list)
            row = {"input_raw": raw_clean, "rxnorm_rxcui": base.get("rxcui"),
                   "rxnorm_name": base.get("rxnorm_name"), "rxnorm_tty": base.get("rxnorm_tty"),
                   **hier}
            if "atc_l1_code" in row:
                row |= _onehots_atc_l1(row["atc_l1_code"])
            rows.append(row)
            new_cache_rows.append({k: row.get(k) for k in cols})

        if use_cache and len(new_cache_rows) >= cache_flush_every:
            _flush_append_csv(cache_csv, pd.DataFrame(new_cache_rows))
            cache = _load_csv(cache_csv, cols)
            cache_idx = {r.lower(): i for i, r in enumerate(cache["input_raw"].astype(str))}
            new_cache_rows = []

        time.sleep(sleep_sec)

    if use_cache and new_cache_rows:
        _flush_append_csv(cache_csv, pd.DataFrame(new_cache_rows))

    return pd.DataFrame(rows)


def clean_medicamentos(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """
    Enriquece 'Nombre del medicamento' con ATC one-hots vía RxNav.
    cache_dir: directorio donde se guarda rxnorm_cache_med.csv
    """
    cache_dir = Path(cache_dir)
    cache_csv = str(cache_dir / "rxnorm_cache_med.csv")

    df_clean = df.copy()
    df_clean["Nombre del medicamento"] = df_clean["Nombre del medicamento"].apply(
        _initial_medicamentos_cleaning
    )

    features = rxnorm_features_for_med_list(
        df_clean["Nombre del medicamento"].tolist(),
        cache_csv=cache_csv,
    )
    atc_l1_cols = [
        "atc_l1_is_A", "atc_l1_is_B", "atc_l1_is_C", "atc_l1_is_D",
        "atc_l1_is_G", "atc_l1_is_H", "atc_l1_is_J", "atc_l1_is_L",
        "atc_l1_is_M", "atc_l1_is_N", "atc_l1_is_R", "atc_l1_is_P",
        "atc_l1_is_S", "atc_l1_is_V",
    ]
    available = [c for c in atc_l1_cols if c in features.columns]
    df_clean = pd.concat([df_clean, features[available].reset_index(drop=True)], axis=1)
    df_clean.drop(columns=["Nombre del medicamento"], inplace=True)
    logger.info("clean_medicamentos OK")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# clean_dx_proc  (GoogleTranslate + ClinicalTables + BART severity)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_query(s: str) -> list:
    s = s or ""
    s = standardize_text(s, replace_char=" ")
    s = remove_stopwords_es(s)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s", ",", s).split(",")


def translate_es_en(text_es: str, cache_df: pd.DataFrame, transl_cache_csv: str,
                    flush_every: int = 50):
    """Traduce ES→EN con cache CSV. Devuelve (text_en, cache_df)."""
    from deep_translator import GoogleTranslator
    _translator = GoogleTranslator(source="es", target="en")

    text_es = _safe(text_es)
    if not text_es:
        return None, cache_df
    idx = {r.lower(): i for i, r in enumerate(cache_df["text_es"].astype(str))} if not cache_df.empty else {}
    key = text_es.lower()
    if key in idx:
        return cache_df.iloc[idx[key]]["text_en"], cache_df
    try:
        text_en = _translator.translate(text_es)
    except Exception:
        text_en = None
    new = pd.DataFrame([{"text_es": text_es, "text_en": text_en}])
    cache_df = pd.concat([cache_df, new], ignore_index=True)
    if len(cache_df) % flush_every == 0:
        _flush_append_csv(transl_cache_csv, cache_df, subset_cols=["text_es"])
        cache_df = _load_csv(transl_cache_csv, ["text_es", "text_en"])
    return text_en, cache_df


def _ct_search(table: str, term: str, maxList: int = 20, other_params=None):
    import itertools
    import requests
    q = _normalize_query(term)
    if not q:
        return []
    url = f"{CT_BASE}/{table}/v3/search"

    def parse_items(data):
        items = []
        for arr in data:
            if isinstance(arr, list):
                for sub in arr:
                    if isinstance(sub, list) and len(sub) >= 2:
                        code, name = sub[0], sub[1]
                        if isinstance(code, str) and isinstance(name, str) and re.match(r"^[A-Z0-9\.]+$", code):
                            items.append({"code": code, "term": name})
        seen = set()
        uniq = []
        for it in items:
            k = (it["code"], it["term"])
            if k not in seen:
                seen.add(k)
                uniq.append(it)
        return uniq

    tried = set()
    attempts = 0
    max_attempts = 5
    for k in range(len(q), 0, -1):
        combs = list(itertools.combinations(q, k))
        combs = sorted(combs, key=lambda x: q.index(x[0]))
        for comb in combs:
            if attempts >= max_attempts:
                break
            key = tuple(comb)
            if key in tried:
                continue
            tried.add(key)
            params = {"terms": ",".join(comb), "maxList": maxList}
            if other_params:
                params.update(other_params)
            try:
                r = requests.get(url, params=params, headers=HEADERS_JSON, timeout=TIMEOUT)
                if r.status_code != 200:
                    attempts += 1
                    continue
                items = parse_items(r.json())
                attempts += 1
                if items:
                    return items
            except Exception:
                attempts += 1
                continue
        if attempts >= max_attempts:
            break
    return []


def _pick_best_ct(items):
    return items[0] if items else {}


def code_diagnoses(series_es: list, cache_dir: Path, use_cache: bool = True,
                   sleep_sec: float = SLEEP_BETWEEN, cache_flush_every: int = 25) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    transl_csv = str(cache_dir / "cache_translate_es_en.csv")
    dx_csv = str(cache_dir / "cache_dx_codes.csv")
    dx_cols = ["input_es", "input_en", "icd10cm_code", "icd10cm_term"]

    transl_cache = _load_csv(transl_csv, ["text_es", "text_en"])
    dx_cache = _load_csv(dx_csv, dx_cols)
    cache_idx = {(row["input_es"] or "").lower(): i
                 for i, row in dx_cache.fillna("").iterrows()} if not dx_cache.empty else {}

    rows = []
    new_cache_rows = []
    for counter, es in enumerate(series_es, 1):
        logger.debug(f"Dx {counter}/{len(series_es)}: {es}")
        if not es or isinstance(es, float):
            rows.append({"input_es": None, "input_en": None,
                         "icd10cm_code": None, "icd10cm_term": None})
            continue
        if use_cache and es.lower() in cache_idx:
            row = dx_cache.iloc[cache_idx[es.lower()]].to_dict()
            rows.append({"input_es": row.get("input_es"), "input_en": row.get("input_en"),
                         "icd10cm_code": row.get("icd10cm_code"),
                         "icd10cm_term": row.get("icd10cm_term")})
            continue
        en, transl_cache = translate_es_en(es, transl_cache, transl_csv)
        ct_en = _ct_search("icd10cm", en, other_params={"sf": "code,name"}) if en else []
        best_ct = _pick_best_ct(ct_en)
        row = {"input_es": es, "input_en": en,
               "icd10cm_code": best_ct.get("code"), "icd10cm_term": best_ct.get("term")}
        rows.append(row)
        new_cache_rows.append(row)
        if use_cache and len(new_cache_rows) >= cache_flush_every:
            _flush_append_csv(dx_csv, pd.DataFrame(new_cache_rows), subset_cols=["input_es"])
            dx_cache = _load_csv(dx_csv, dx_cols)
            cache_idx = {(r["input_es"] or "").lower(): i
                         for i, r in dx_cache.fillna("").iterrows()}
            new_cache_rows = []
        time.sleep(sleep_sec)

    _flush_append_csv(transl_csv, transl_cache, subset_cols=["text_es"])
    if use_cache and new_cache_rows:
        _flush_append_csv(dx_csv, pd.DataFrame(new_cache_rows), subset_cols=["input_es"])
    return pd.DataFrame(rows)


def code_procedures(series_es: list, cache_dir: Path, cache_filename: str,
                    use_cache: bool = True, sleep_sec: float = SLEEP_BETWEEN,
                    cache_flush_every: int = 25) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    transl_csv = str(cache_dir / "cache_translate_es_en.csv")
    proc_csv = str(cache_dir / cache_filename)
    proc_cols = ["input_es", "input_en", "icd10pcs_code", "icd10pcs_term"]

    transl_cache = _load_csv(transl_csv, ["text_es", "text_en"])
    proc_cache = _load_csv(proc_csv, proc_cols)
    cache_idx = {(row["input_es"] or "").lower(): i
                 for i, row in proc_cache.fillna("").iterrows()} if not proc_cache.empty else {}

    rows = []
    new_cache_rows = []
    for counter, es in enumerate(series_es, 1):
        logger.debug(f"Proc {counter}/{len(series_es)}: {es}")
        if not es or isinstance(es, float):
            rows.append({"input_es": None, "input_en": None,
                         "icd10pcs_code": None, "icd10pcs_term": None})
            continue
        if use_cache and es.lower() in cache_idx:
            row = proc_cache.iloc[cache_idx[es.lower()]].to_dict()
            rows.append({"input_es": row.get("input_es"), "input_en": row.get("input_en"),
                         "icd10pcs_code": row.get("icd10pcs_code"),
                         "icd10pcs_term": row.get("icd10pcs_term")})
            continue
        en, transl_cache = translate_es_en(es, transl_cache, transl_csv)
        ct_en = _ct_search("hcpcs", en) if en else []
        best_ct = _pick_best_ct(ct_en)
        row = {"input_es": es, "input_en": en,
               "icd10pcs_code": best_ct.get("code"), "icd10pcs_term": best_ct.get("term")}
        rows.append(row)
        new_cache_rows.append(row)
        if use_cache and len(new_cache_rows) >= cache_flush_every:
            _flush_append_csv(proc_csv, pd.DataFrame(new_cache_rows), subset_cols=["input_es"])
            proc_cache = _load_csv(proc_csv, proc_cols)
            cache_idx = {(r["input_es"] or "").lower(): i
                         for i, r in proc_cache.fillna("").iterrows()}
            new_cache_rows = []
        time.sleep(sleep_sec)

    _flush_append_csv(transl_csv, transl_cache, subset_cols=["text_es"])
    if use_cache and new_cache_rows:
        _flush_append_csv(proc_csv, pd.DataFrame(new_cache_rows), subset_cols=["input_es"])
    return pd.DataFrame(rows)


def _get_code(text):
    if not text or isinstance(text, float):
        return None
    if text == "NO ENCONTRADO":
        return "NO ENCONTRADO"
    return text[0]


def _hash_text(text: str) -> str:
    if pd.isna(text) or isinstance(text, float):
        return ""
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def _merge_severity_results(df_clean: pd.DataFrame, text_column: str,
                             results_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if results_df.empty:
        return df_clean
    hash_col = f"_temp_hash{suffix}"
    df_clean[hash_col] = df_clean[text_column].apply(_hash_text)
    results_unique = results_df.drop_duplicates(subset=["text_hash"], keep="first").copy()
    results_unique = results_unique.add_suffix(suffix)
    text_hash_col = f"text_hash{suffix}"
    df_clean = df_clean.merge(results_unique, left_on=hash_col,
                              right_on=text_hash_col, how="left")
    df_clean.drop(columns=[hash_col], inplace=True)
    return df_clean


def _separate_columns(df: pd.DataFrame, all_scores_col: str, all_labels_col: str,
                      prefix: str, labels_universo: list = None) -> pd.DataFrame:
    df_clean = df.copy()

    def safe_parse(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        return x if isinstance(x, list) else []

    scores_series = df_clean[all_scores_col].apply(safe_parse)
    labels_series = df_clean[all_labels_col].apply(safe_parse)

    if labels_universo is None:
        all_unique = set()
        for l_list in labels_series:
            all_unique.update(l_list)
        labels_universo = sorted(list(all_unique))

    aligned_data = []
    for scores, labels in zip(scores_series, labels_series):
        mapping = dict(zip(labels, scores))
        row_scores = [mapping.get(label, 0.0) for label in labels_universo]
        aligned_data.append(row_scores)

    new_col_names = [f"score_{prefix}_{l.replace(' ', '_')}" for l in labels_universo]
    df_scores = pd.DataFrame(aligned_data, columns=new_col_names, index=df_clean.index)
    df_clean = pd.concat([df_clean, df_scores], axis=1)
    df_clean.drop(columns=[all_scores_col, all_labels_col], inplace=True)
    return df_clean


def _encode_dx_proc_ant(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """
    Codifica con onehot los códigos ICD-10 y aplica BART-MNLI severity.
    Migrado fielmente de encode_dx_proc_ant() en notebook cell 34.
    """
    from src.utils.severity_classification import process_list_of_texts

    cache_dir = Path(cache_dir)
    df_clean = df.copy()

    df_clean["Dx Preoperatorio Code"] = df_clean["Dx Preoperatorio Code"].apply(_get_code)
    df_clean["Procedimiento propuesto Code"] = df_clean["Procedimiento propuesto Code"].apply(_get_code)
    df_clean["Antecedentes quirúrgicos Code"] = df_clean["Antecedentes quirúrgicos Code"].apply(_get_code)

    df_clean = encode_onehot(df_clean, "Dx Preoperatorio Code")
    df_clean = encode_onehot(df_clean, "Procedimiento propuesto Code")
    df_clean = encode_onehot(df_clean, "Antecedentes quirúrgicos Code")

    results_dx = process_list_of_texts(
        df_clean["Dx Preoperatorio EN"].tolist(),
        cache_path=str(cache_dir / "cache_dx_codes_results.csv"),
        append_every=100,
    )
    results_proc = process_list_of_texts(
        df_clean["Procedimiento propuesto EN"].tolist(),
        cache_path=str(cache_dir / "cache_proc_codes_results.csv"),
        append_every=100,
    )
    results_proc_ant = process_list_of_texts(
        df_clean["Antecedentes quirúrgicos EN"].tolist(),
        cache_path=str(cache_dir / "cache_proc_ant_codes_results.csv"),
        append_every=100,
    )

    df_clean = df_clean.reset_index(drop=True)
    df_clean = _merge_severity_results(df_clean, "Dx Preoperatorio EN", results_dx, "_dx")
    df_clean = _merge_severity_results(df_clean, "Procedimiento propuesto EN", results_proc, "_proc")
    df_clean = _merge_severity_results(df_clean, "Antecedentes quirúrgicos EN", results_proc_ant, "_proc_ant")

    severity_classes = ["NONE", "low severity", "moderate severity",
                        "medium severity", "high severity", "critical"]

    for col in ("predicted_label_dx", "predicted_label_proc", "predicted_label_proc_ant"):
        df_clean[col] = df_clean[col].fillna("NONE")

    df_clean = encode_label_with_classes(df_clean, "predicted_label_dx", severity_classes)
    df_clean = encode_label_with_classes(df_clean, "predicted_label_proc", severity_classes)
    df_clean = encode_label_with_classes(df_clean, "predicted_label_proc_ant", severity_classes)

    df_clean = _separate_columns(df_clean, "all_scores_dx", "all_labels_dx", "dx")
    df_clean = _separate_columns(df_clean, "all_scores_proc", "all_labels_proc", "proc")
    df_clean = _separate_columns(df_clean, "all_scores_proc_ant", "all_labels_proc_ant", "proc_ant")

    df_clean.drop(columns=["Dx Preoperatorio Code", "Procedimiento propuesto Code",
                            "Antecedentes quirúrgicos Code"], inplace=True)
    df_clean.drop(columns=["predicted_score_dx", "predicted_score_proc",
                            "predicted_score_proc_ant"], inplace=True)
    return df_clean


def clean_dx_proc(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """
    Enriquece Dx Preoperatorio, Procedimiento propuesto y Antecedentes quirúrgicos
    con códigos ICD-10 (ClinicalTables) y severidad clínica (BART-MNLI).
    cache_dir: directorio donde se guardan los CSV cache de traducción y codificación.
    """
    cache_dir = Path(cache_dir)
    df_clean = df.copy()

    df_dx = code_diagnoses(df_clean["Dx Preoperatorio"].tolist(),
                            cache_dir=cache_dir, use_cache=True)
    df_proc = code_procedures(df_clean["Procedimiento propuesto"].tolist(),
                               cache_dir=cache_dir,
                               cache_filename="cache_proc_codes.csv", use_cache=True)
    df_proc_ant = code_procedures(df_clean["Antecedentes quirúrgicos"].tolist(),
                                   cache_dir=cache_dir,
                                   cache_filename="cache_proc_ant_codes.csv", use_cache=True)

    df_dx.rename(columns={
        "input_en": "Dx Preoperatorio EN",
        "icd10cm_code": "Dx Preoperatorio Code",
        "icd10cm_term": "Dx Preoperatorio Term",
    }, inplace=True)
    df_proc.rename(columns={
        "input_en": "Procedimiento propuesto EN",
        "icd10pcs_code": "Procedimiento propuesto Code",
        "icd10pcs_term": "Procedimiento propuesto Term",
    }, inplace=True)
    df_proc_ant.rename(columns={
        "input_en": "Antecedentes quirúrgicos EN",
        "icd10pcs_code": "Antecedentes quirúrgicos Code",
        "icd10pcs_term": "Antecedentes quirúrgicos Term",
    }, inplace=True)

    df_clean = pd.concat([
        df_clean,
        df_dx[["Dx Preoperatorio EN", "Dx Preoperatorio Code", "Dx Preoperatorio Term"]],
        df_proc[["Procedimiento propuesto EN", "Procedimiento propuesto Code", "Procedimiento propuesto Term"]],
        df_proc_ant[["Antecedentes quirúrgicos EN", "Antecedentes quirúrgicos Code", "Antecedentes quirúrgicos Term"]],
    ], axis=1)

    # Marcar "NO ENCONTRADO" donde el texto fuente existe pero no se encontró código
    for src, tgt in [
        ("Dx Preoperatorio", "Dx Preoperatorio EN"),
        ("Dx Preoperatorio", "Dx Preoperatorio Code"),
        ("Dx Preoperatorio", "Dx Preoperatorio Term"),
        ("Procedimiento propuesto", "Procedimiento propuesto EN"),
        ("Procedimiento propuesto", "Procedimiento propuesto Code"),
        ("Procedimiento propuesto", "Procedimiento propuesto Term"),
        ("Antecedentes quirúrgicos", "Antecedentes quirúrgicos EN"),
        ("Antecedentes quirúrgicos", "Antecedentes quirúrgicos Code"),
        ("Antecedentes quirúrgicos", "Antecedentes quirúrgicos Term"),
    ]:
        mask = df_clean[src].notnull() & df_clean[tgt].isnull()
        df_clean.loc[mask, tgt] = "NO ENCONTRADO"

    df_clean = _encode_dx_proc_ant(df_clean, cache_dir)

    logger.info("clean_dx_proc OK")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada: aplica clean_medicamentos + clean_dx_proc juntos
# ─────────────────────────────────────────────────────────────────────────────

def enrich_preop(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """
    Aplica las dos etapas de enriquecimiento basadas en APIs externas:
      1. clean_medicamentos (RxNav ATC)
      2. clean_dx_proc (ClinicalTables + BART)

    Se llama desde task_clean_data DESPUÉS de clean_preop().
    """
    df_out = df.copy()
    if "Nombre del medicamento" in df_out.columns:
        df_out = clean_medicamentos(df_out, cache_dir)
    if "Dx Preoperatorio" in df_out.columns:
        df_out = clean_dx_proc(df_out, cache_dir)
    return df_out
