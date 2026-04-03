# dags/preanesthesia_pipeline.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ruta al proyecto dentro del contenedor (configurada como volumen en docker-compose)
PROJECT_ROOT = Path("/opt/airflow/project")
CONFIG_DIR = PROJECT_ROOT / "config"

# Cargar config al inicializar el DAG (una sola vez)
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config import load_config

cfg = load_config(CONFIG_DIR)
ACTIVE_TARGETS = cfg.active_targets
ENABLED_MODELS = list(cfg.enabled_models().keys())


def _placeholder(task_name: str, **kwargs):
    """Placeholder hasta que se implemente el módulo correspondiente."""
    import logging
    logging.getLogger(f"pipeline.{task_name}").info(f"Task {task_name} — pendiente de implementación")


with DAG(
    dag_id="preanesthesia_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,           # Solo ejecución manual
    catchup=False,
    tags=["screening", "preanesthesia"],
) as dag:

    # ── Paso 1: Validación ────────────────────────────────────────────────
    validate_raw = PythonOperator(
        task_id="validate_raw_data",
        python_callable=_placeholder,
        op_kwargs={"task_name": "validate_raw_data"},
    )

    eda_preop_raw = PythonOperator(
        task_id="eda_preop_raw",
        python_callable=_placeholder,
        op_kwargs={"task_name": "eda_preop_raw"},
    )

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_placeholder,
        op_kwargs={"task_name": "clean_data"},
    )

    eda_preop_clean = PythonOperator(
        task_id="eda_preop_clean",
        python_callable=_placeholder,
        op_kwargs={"task_name": "eda_preop_clean"},
    )

    # ── Pasos por target (paralelos) ──────────────────────────────────────
    extract_target_tasks = {}
    merge_tasks = {}
    eda_posop_tasks = {}
    select_features_tasks = {}
    eda_correlation_tasks = {}
    train_tasks = {}
    evaluate_tasks = {}
    plot_tasks = {}

    for target in ACTIVE_TARGETS:
        extract_target_tasks[target] = PythonOperator(
            task_id=f"extract_target__{target}",
            python_callable=_placeholder,
            op_kwargs={"task_name": f"extract_target__{target}"},
        )
        merge_tasks[target] = PythonOperator(
            task_id=f"merge_datasets__{target}",
            python_callable=_placeholder,
            op_kwargs={"task_name": f"merge_datasets__{target}"},
        )
        eda_posop_tasks[target] = PythonOperator(
            task_id=f"eda_posop__{target}",
            python_callable=_placeholder,
            op_kwargs={"task_name": f"eda_posop__{target}"},
        )
        select_features_tasks[target] = PythonOperator(
            task_id=f"select_features__{target}",
            python_callable=_placeholder,
            op_kwargs={"task_name": f"select_features__{target}"},
        )
        eda_correlation_tasks[target] = PythonOperator(
            task_id=f"eda_correlation__{target}",
            python_callable=_placeholder,
            op_kwargs={"task_name": f"eda_correlation__{target}"},
        )

        for model in ENABLED_MODELS:
            key = f"{model}__{target}"
            train_tasks[key] = PythonOperator(
                task_id=f"train__{model}__{target}",
                python_callable=_placeholder,
                op_kwargs={"task_name": f"train__{model}__{target}"},
            )
            evaluate_tasks[key] = PythonOperator(
                task_id=f"evaluate__{model}__{target}",
                python_callable=_placeholder,
                op_kwargs={"task_name": f"evaluate__{model}__{target}"},
            )
            plot_tasks[key] = PythonOperator(
                task_id=f"model_plots__{model}__{target}",
                python_callable=_placeholder,
                op_kwargs={"task_name": f"model_plots__{model}__{target}"},
            )

    comparison_report = PythonOperator(
        task_id="generate_comparison_report",
        python_callable=_placeholder,
        op_kwargs={"task_name": "generate_comparison_report"},
    )

    # ── Dependencias ──────────────────────────────────────────────────────
    validate_raw >> eda_preop_raw >> clean_data >> eda_preop_clean

    for target in ACTIVE_TARGETS:
        eda_preop_clean >> extract_target_tasks[target]
        extract_target_tasks[target] >> merge_tasks[target]
        merge_tasks[target] >> eda_posop_tasks[target]
        merge_tasks[target] >> select_features_tasks[target]
        [eda_posop_tasks[target], select_features_tasks[target]] >> eda_correlation_tasks[target]

        for model in ENABLED_MODELS:
            key = f"{model}__{target}"
            eda_correlation_tasks[target] >> train_tasks[key]
            train_tasks[key] >> evaluate_tasks[key]
            evaluate_tasks[key] >> plot_tasks[key]
            plot_tasks[key] >> comparison_report
