"""
Módulo para extracción de variables medibles de textos médicos usando SOLO modelos de IA
Compara resultados entre múltiples modelos médicos (zero-shot con paralelismo + caché)
"""

import os
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Set

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline
import torch

# Detectar GPU (si no hay, seguirá en CPU)
DEVICE = 0 if torch.cuda.is_available() else -1

# Modelos zero-shot (ajusta según tus necesidades; distilados son más rápidos)
ZERO_SHOT_MODELS: Dict[str, str] = {
    "facebook_bart": "facebook/bart-large-mnli"
    # Alternativas más ligeras:
    # "distilroberta": "typeform/distilroberta-base-mnli",
    # "distilbart": "valhalla/distilbart-mnli-12-3",
}

# Configuración de paralelismo
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Ajusta según tu CPU
DEFAULT_BATCH_SIZE = 32

# Caché global de pipelines por modelo (compartido entre hilos)
MODEL_PIPELINES: Dict[str, pipeline] = {}
MODEL_LOCK = threading.Lock()


def _get_pipeline(model_name: str, model_id: str):
    """Obtiene (y cachea) el pipeline zero-shot para un modelo concreto."""
    if model_name in MODEL_PIPELINES:
        return MODEL_PIPELINES[model_name]

    with MODEL_LOCK:
        if model_name not in MODEL_PIPELINES:
            MODEL_PIPELINES[model_name] = pipeline(
                "zero-shot-classification",
                model=model_id,
                device=DEVICE,
                framework="pt",
            )
    return MODEL_PIPELINES[model_name]


def _hash_text(text: str) -> str:
    """Genera un hash estable para identificar textos procesados."""
    # Puede ser NaN entonces: AttributeError: 'float' object has no attribute 'encode'
    if pd.isna(text) or isinstance(text, float):
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_valid_text(text) -> bool:
    """Verifica si un texto es válido para clasificar."""
    if text is None:
        return False
    if pd.isna(text):
        return False
    if isinstance(text, float):
        return False
    text_str = str(text).strip()
    if text_str == '' or text_str.lower() == 'none':
        return False
    return True

def _classify_batch(
    model_name: str,
    model_id: str,
    texts: List[str],
    candidate_labels: List[str]
):
    """Clasifica un lote de textos con un modelo concreto."""
    # Filtrar textos válidos (no None, no vacíos)
    valid_texts = [t for t in texts if _is_valid_text(t)]
    
    if not valid_texts:
        return []
    
    classifier = _get_pipeline(model_name, model_id)
    outputs = classifier(valid_texts, candidate_labels, batch_size=len(valid_texts))
    if isinstance(outputs, dict):  # lote de tamaño 1
        outputs = [outputs]

    rows = []
    for text, output in zip(valid_texts, outputs):
        rows.append({
            "model": model_name,
            "text": text,
            "text_hash": _hash_text(text),
            "predicted_label": output["labels"][0],
            "predicted_score": output["scores"][0],
            "all_labels": output["labels"],
            "all_scores": output["scores"],
            "status": "success",
        })
    return rows


def _load_cache(cache_path: Path) -> Tuple[pd.DataFrame, Set[Tuple[str, str]]]:
    """Carga resultados previos y devuelve DF + conjunto de (hash, modelo) procesados."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if "text_hash" not in df.columns:
            df["text_hash"] = df["text"].apply(_hash_text)
        processed = set(zip(df["text_hash"], df["model"]))
        return df, processed
    return pd.DataFrame(), set()


def _append_results(cache_path: Path, rows: List[dict]):
    """Escribe filas en modo append y añade hash si falta."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "text_hash" not in df.columns:
        df["text_hash"] = df["text"].apply(_hash_text)
    header = not cache_path.exists()
    df.to_csv(cache_path, mode="a", header=header, index=False)


def zero_shot_severity_classification(
    texts: List[str],
    candidate_labels: List[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    models: Dict[str, str] = None,
    cache_path: str | None = None,
    append_every: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clasifica una lista de textos con zero-shot usando varios hilos y caché incremental.

    - `texts`: lista de diagnósticos/procedimientos.
    - `candidate_labels`: etiquetas candidatas (por defecto severidad básica).
    - `batch_size`: tamaño del lote.
    - `max_workers`: número de hilos.
    - `models`: diccionario {nombre_modelo: modelo_hf}.
    - `cache_path`: ruta CSV para caché (si None, sin caché).
    - `append_every`: guarda en CSV después de acumular este número de filas nuevas.
    - `verbose`: si True, muestra progreso.

    Devuelve un DataFrame con columnas:
    ['model', 'text', 'text_hash', 'predicted_label', 'predicted_score',
     'all_labels', 'all_scores', 'status'].
    """
    if not texts:
        if verbose:
            print(f"  ℹ️ No hay textos para procesar")
        return pd.DataFrame()

    candidate_labels = candidate_labels or [
        "low severity", "moderate severity", "medium severity", "high severity", "critical"
    ]
    models = models or ZERO_SHOT_MODELS

    cache_df, processed_pairs = (pd.DataFrame(), set())
    cache_file = Path(cache_path) if cache_path else None

    # Cargar cache previa si existe
    if cache_file:
        cache_df, processed_pairs = _load_cache(cache_file)

    # Filtrar textos pendientes (no presentes para todos los modelos Y que sean válidos)
    pending_texts = []
    for text in texts:
        # Saltar textos inválidos (None, NaN, vacíos)
        if not _is_valid_text(text):
            continue
        text_hash = _hash_text(text)
        # ¿ya está procesado para todos los modelos?
        if all((text_hash, model_name) in processed_pairs for model_name in models):
            continue
        pending_texts.append(text)

    if not pending_texts:
        if verbose:
            print(f"  ℹ️ No hay textos pendientes por procesar")
        return cache_df.copy() if not cache_df.empty else pd.DataFrame()

    if verbose:
        print(f"  📝 Textos pendientes: {len(pending_texts)}")

    batches = [pending_texts[i:i + batch_size] for i in range(0, len(pending_texts), batch_size)]
    tasks = []
    new_rows = []

    total_to_process = len(pending_texts) * len(models)
    processed_count = 0
    
    if verbose:
        print(f"  🔄 Iniciando procesamiento de {total_to_process} textos en {len(batches)} lotes...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_name, model_id in models.items():
            for batch in batches:
                # Determinar qué textos de este lote faltan para este modelo
                texts_to_process = [
                    text for text in batch
                    if (_hash_text(text), model_name) not in processed_pairs
                ]
                if not texts_to_process:
                    continue

                tasks.append(executor.submit(
                    _classify_batch,
                    model_name,
                    model_id,
                    texts_to_process,
                    candidate_labels,
                ))

        for future in as_completed(tasks):
            task_rows = []
            try:
                task_rows = future.result()
            except Exception as exc:
                print(f"❌ ERROR durante clasificación: {exc}")
                import traceback
                traceback.print_exc()
                task_rows = []
            if not task_rows:
                continue

            new_rows.extend(task_rows)
            processed_count += len(task_rows)
            if verbose:
                print(f"Procesados: {processed_count}/{total_to_process}")

            if cache_file and len(new_rows) >= append_every:
                _append_results(cache_file, new_rows)
                for row in new_rows:
                    processed_pairs.add((row["text_hash"], row["model"]))
                new_rows = []

    if cache_file and new_rows:
        _append_results(cache_file, new_rows)
        for row in new_rows:
            processed_pairs.add((row["text_hash"], row["model"]))
        if verbose:
            print(f"Procesados: {processed_count}/{total_to_process}")

    result_df = pd.DataFrame(new_rows)
    if not cache_df.empty:
        result_df = pd.concat([cache_df, result_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset=["text_hash", "model"], keep="last")

    return result_df


def process_list_of_texts(
    texts: List[str],
    candidate_labels: List[str] = ["low severity", "moderate severity", "medium severity", "high severity", "critical"],
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    models: Dict[str, str] = None,
    cache_path: str | None = None,
    append_every: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Wrapper conveniente para procesar una lista completa, con contador de progreso."""
    if verbose:
        print(f"  ℹ️ Procesando {len(texts)} textos")
    return zero_shot_severity_classification(
        texts,
        candidate_labels=candidate_labels,
        batch_size=batch_size,
        max_workers=max_workers,
        models=models,
        cache_path=cache_path,
        append_every=append_every,
        verbose=verbose,
    )