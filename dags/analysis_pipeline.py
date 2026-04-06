# dags/analysis_pipeline.py
"""
DAG independiente para análisis exploratorios posoperatorios.
Corre manualmente (schedule=None) desde la UI de Airflow.

Tareas:
  1. flag_predictability  — ¿qué flags son predecibles desde preop? (Enfoque C)
  2. error_subgroups      — ¿dónde falla el modelo? (Enfoque B)
  3. posop_clustering     — ¿qué estructuras internas hay? (Enfoque A)

Las tres tareas son independientes y corren en paralelo.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path("/opt/airflow/project")
CONFIG_DIR = PROJECT_ROOT / "config"

sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_config
from src.utils.logger import get_logger

cfg = load_config(CONFIG_DIR)
logger = get_logger("dag.analysis")

# Paths fijos para este análisis — usa siempre target_d_v2_hosp y random_forest
_VERSION = cfg.output_path()  # output/v1
_PROC_DIR = _VERSION / "data_processed"
_TARGET = "target_d_v2_hosp"
_MODEL_KEY = "random_forest"
_MERGED_PATH = _PROC_DIR / _TARGET / "merged.parquet"
_POSOP_PATH = _PROC_DIR / "posop_raw.parquet"
_SPLITS_DIR = _PROC_DIR / _TARGET / "splits"
_MODEL_PATH = _VERSION / "models" / _TARGET / f"{_MODEL_KEY}_model.joblib"
_METRICS_PATH = _VERSION / "models" / _TARGET / f"{_MODEL_KEY}_metrics.json"
_OUTPUT_DIR = _VERSION / "reports" / "analisis_posoperatorio"


def task_flag_predictability(**kwargs):
    from src.analysis.flag_predictability import run_flag_predictability
    run_flag_predictability(
        merged_path=_MERGED_PATH,
        output_dir=_OUTPUT_DIR,
        min_prevalence=0.01,
        n_folds=5,
        random_state=cfg.train_test_split.get("random_state", 42),
    )


def task_error_subgroups(**kwargs):
    import json
    from src.analysis.error_subgroups import run_error_subgroups

    with open(_METRICS_PATH) as f:
        threshold = json.load(f).get("Threshold", 0.17)

    run_error_subgroups(
        merged_path=_MERGED_PATH,
        splits_dir=_SPLITS_DIR,
        model_path=_MODEL_PATH,
        output_dir=_OUTPUT_DIR,
        threshold=threshold,
    )


def task_posop_clustering(**kwargs):
    from src.analysis.posop_clustering import run_posop_clustering
    run_posop_clustering(
        posop_path=_POSOP_PATH,
        merged_path=_MERGED_PATH,
        output_dir=_OUTPUT_DIR,
        n_clusters=5,
        random_state=cfg.train_test_split.get("random_state", 42),
    )


with DAG(
    dag_id="analysis_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["screening", "preanesthesia", "analysis"],
) as dag:

    flag_pred = PythonOperator(
        task_id="flag_predictability",
        python_callable=task_flag_predictability,
    )
    error_sub = PythonOperator(
        task_id="error_subgroups",
        python_callable=task_error_subgroups,
    )
    clustering = PythonOperator(
        task_id="posop_clustering",
        python_callable=task_posop_clustering,
    )

    # Las tres tareas son independientes — corren en paralelo
    [flag_pred, error_sub, clustering]
