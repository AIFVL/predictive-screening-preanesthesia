# dags/preanesthesia_pipeline.py
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
ACTIVE_TARGETS = cfg.active_targets
ENABLED_MODELS = list(cfg.enabled_models().keys())
logger = get_logger("dag")


# ── Callables reales ──────────────────────────────────────────────────────────

def task_validate_raw_data(**kwargs):
    from src.data.loader import load_raw_data
    from src.data.validator import validate_raw_data

    raw_dir = PROJECT_ROOT / cfg.raw_data_path()
    out_dir = cfg.output_path() / "data_processed"

    df_pre, df_pos = load_raw_data(raw_dir, out_dir)
    report = validate_raw_data(df_pre, df_pos, cfg.output_path() / "reports")
    if report["warnings"]:
        logger.warning(f"Validación con advertencias: {report['warnings']}")


def task_eda_preop_raw(**kwargs):
    from src.data.loader import load_raw_data
    from src.reports.eda import generate_preop_eda

    raw_dir = PROJECT_ROOT / cfg.raw_data_path()
    out_dir = cfg.output_path() / "data_processed"
    df_pre, _ = load_raw_data(raw_dir, out_dir)
    generate_preop_eda(df_pre, cfg.output_path() / "plots", label="eda_preop_raw")


def task_clean_data(**kwargs):
    from src.data.loader import load_raw_data
    from src.cleaning.cleaner import clean_preop
    from src.cleaning.report import generate_cleaning_report
    from src.utils.io import write_parquet

    raw_dir = PROJECT_ROOT / cfg.raw_data_path()
    proc_dir = cfg.output_path() / "data_processed"
    df_pre, _ = load_raw_data(raw_dir, proc_dir)
    df_clean = clean_preop(df_pre, cfg.cleaning)
    write_parquet(df_clean, proc_dir / "cleaned.parquet")
    generate_cleaning_report(df_pre, df_clean, cfg.output_path() / "reports")


def task_eda_preop_clean(**kwargs):
    from src.utils.io import read_parquet
    from src.reports.eda import generate_preop_eda

    proc_dir = cfg.output_path() / "data_processed"
    df_clean = read_parquet(proc_dir / "cleaned.parquet")
    generate_preop_eda(df_clean, cfg.output_path() / "plots", label="eda_preop_clean")


def make_task_extract_target(target_name: str):
    def _task(**kwargs):
        from src.data.loader import load_raw_data
        from src.target.pipeline import run_target_extraction
        from src.utils.io import write_parquet

        raw_dir = PROJECT_ROOT / cfg.raw_data_path()
        proc_dir = cfg.output_path() / "data_processed"
        _, df_pos = load_raw_data(raw_dir, proc_dir)
        target_cfg = cfg.get_target(target_name)
        df_result = run_target_extraction(df_pos, target_cfg)
        write_parquet(df_result, proc_dir / target_name / "target_extracted.parquet")
    _task.__name__ = f"task_extract_target_{target_name}"
    return _task


def make_task_merge_datasets(target_name: str):
    def _task(**kwargs):
        from src.utils.io import read_parquet, write_parquet
        from src.datasets.merge import merge_preop_target

        proc_dir = cfg.output_path() / "data_processed"
        df_clean = read_parquet(proc_dir / "cleaned.parquet")
        df_target = read_parquet(proc_dir / target_name / "target_extracted.parquet")
        df_merged = merge_preop_target(df_clean, df_target)
        write_parquet(df_merged, proc_dir / target_name / "merged.parquet")
    _task.__name__ = f"task_merge_datasets_{target_name}"
    return _task


def make_task_eda_posop(target_name: str):
    def _task(**kwargs):
        from src.utils.io import read_parquet
        from src.reports.eda import generate_posop_eda

        proc_dir = cfg.output_path() / "data_processed"
        df_merged = read_parquet(proc_dir / target_name / "merged.parquet")
        generate_posop_eda(df_merged, cfg.output_path() / "plots", target_name)
    _task.__name__ = f"task_eda_posop_{target_name}"
    return _task


def make_task_select_features(target_name: str):
    def _task(**kwargs):
        from src.utils.io import read_parquet, write_parquet, write_json
        from src.features.selection import rank_and_select_features
        from src.datasets.splits import stratified_train_test_split

        proc_dir = cfg.output_path() / "data_processed"
        df_merged = read_parquet(proc_dir / target_name / "merged.parquet")

        ranking, selected_features, metadata = rank_and_select_features(
            df_merged,
            version_name=target_name,
            encoding_fix_map=cfg.features.get("encoding_fix_map"),
            random_state=cfg.train_test_split.get("random_state", 42),
        )

        features_path = proc_dir / target_name / "selected_features.json"
        write_json({"features": selected_features, "metadata": metadata}, features_path)

        X = df_merged[selected_features].copy()
        y = df_merged["target"].astype(int)
        splits_dir = proc_dir / target_name / "splits"
        stratified_train_test_split(
            X, y,
            test_size=cfg.train_test_split.get("test_size", 0.2),
            random_state=cfg.train_test_split.get("random_state", 42),
            out_dir=splits_dir,
        )
    _task.__name__ = f"task_select_features_{target_name}"
    return _task


def make_task_eda_correlation(target_name: str):
    def _task(**kwargs):
        from src.utils.io import read_parquet
        from src.reports.correlation import analyze_preop_posop_correlation

        proc_dir = cfg.output_path() / "data_processed"
        df_merged = read_parquet(proc_dir / target_name / "merged.parquet")
        analyze_preop_posop_correlation(
            df_merged, target_name, cfg.output_path() / "plots"
        )
    _task.__name__ = f"task_eda_correlation_{target_name}"
    return _task


def _placeholder_model(task_name: str, **kwargs):
    logger.info(f"Task {task_name} — pendiente (Plan 3)")


# ── DAG ──────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="preanesthesia_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["screening", "preanesthesia"],
) as dag:

    validate_raw = PythonOperator(task_id="validate_raw_data", python_callable=task_validate_raw_data)
    eda_preop_raw = PythonOperator(task_id="eda_preop_raw", python_callable=task_eda_preop_raw)
    clean_data = PythonOperator(task_id="clean_data", python_callable=task_clean_data)
    eda_preop_clean = PythonOperator(task_id="eda_preop_clean", python_callable=task_eda_preop_clean)

    extract_target_tasks: dict = {}
    merge_tasks: dict = {}
    eda_posop_tasks: dict = {}
    select_features_tasks: dict = {}
    eda_correlation_tasks: dict = {}
    train_tasks: dict = {}
    evaluate_tasks: dict = {}
    plot_tasks: dict = {}

    for target in ACTIVE_TARGETS:
        extract_target_tasks[target] = PythonOperator(
            task_id=f"extract_target__{target}",
            python_callable=make_task_extract_target(target),
        )
        merge_tasks[target] = PythonOperator(
            task_id=f"merge_datasets__{target}",
            python_callable=make_task_merge_datasets(target),
        )
        eda_posop_tasks[target] = PythonOperator(
            task_id=f"eda_posop__{target}",
            python_callable=make_task_eda_posop(target),
        )
        select_features_tasks[target] = PythonOperator(
            task_id=f"select_features__{target}",
            python_callable=make_task_select_features(target),
        )
        eda_correlation_tasks[target] = PythonOperator(
            task_id=f"eda_correlation__{target}",
            python_callable=make_task_eda_correlation(target),
        )

        for model in ENABLED_MODELS:
            key = f"{model}__{target}"
            train_tasks[key] = PythonOperator(
                task_id=f"train__{model}__{target}",
                python_callable=_placeholder_model,
                op_kwargs={"task_name": f"train__{model}__{target}"},
            )
            evaluate_tasks[key] = PythonOperator(
                task_id=f"evaluate__{model}__{target}",
                python_callable=_placeholder_model,
                op_kwargs={"task_name": f"evaluate__{model}__{target}"},
            )
            plot_tasks[key] = PythonOperator(
                task_id=f"model_plots__{model}__{target}",
                python_callable=_placeholder_model,
                op_kwargs={"task_name": f"model_plots__{model}__{target}"},
            )

    comparison_report = PythonOperator(
        task_id="generate_comparison_report",
        python_callable=_placeholder_model,
        op_kwargs={"task_name": "generate_comparison_report"},
    )

    # ── Dependencias ─────────────────────────────────────────────────────────
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
