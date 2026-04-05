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
    from src.cleaning.enrichment import enrich_preop
    from src.cleaning.report import generate_cleaning_report
    from src.utils.io import write_parquet

    raw_dir = PROJECT_ROOT / cfg.raw_data_path()
    proc_dir = cfg.output_path() / "data_processed"
    cache_dir = PROJECT_ROOT / "cache"
    df_pre, _ = load_raw_data(raw_dir, proc_dir)

    # Etapas deterministas (sin APIs)
    df_clean = clean_preop(df_pre, cfg.cleaning)

    # Etapas de enriquecimiento con APIs externas (medicamentos, dx/proc, severidad)
    df_clean = enrich_preop(df_clean, cache_dir)

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


def make_task_train_model(target_name: str, model_key: str):
    def _task(**kwargs):
        from src.utils.io import read_parquet
        from src.models.trainer import train_model
        from src.models.hyperparameter_search import search_hyperparameters

        proc_dir = cfg.output_path() / "data_processed"
        splits_dir = proc_dir / target_name / "splits"

        X_train = read_parquet(splits_dir / "X_train.parquet")
        y_train = read_parquet(splits_dir / "y_train.parquet")["target"]

        all_models = cfg.enabled_models()
        model_cfg = dict(all_models[model_key])
        hp_cfg = cfg.hyperparameter_search

        # Resolve sub-estimator configs for ensemble models
        if "estimators" in model_cfg:
            model_cfg["_estimator_configs"] = {
                name: all_models[name] for name in model_cfg["estimators"]
            }
            if "meta_estimator" in model_cfg:
                meta_name = model_cfg["meta_estimator"]
                model_cfg["_estimator_configs"][meta_name] = all_models[meta_name]

        if hp_cfg.get("enabled", False):
            best_params = search_hyperparameters(
                model_cfg, X_train, y_train,
                n_iter=hp_cfg.get("n_iter", 70),
                scoring=hp_cfg.get("scoring", "f2"),
                cv_folds=hp_cfg.get("cv_folds", 5),
                random_state=cfg.train_test_split.get("random_state", 42),
            )
            model_cfg = {**model_cfg, "params": best_params}

        out_dir = cfg.output_path() / "models" / target_name
        train_model(
            model_cfg, X_train, y_train,
            out_dir=out_dir,
            model_name=model_key,
            random_state=cfg.train_test_split.get("random_state", 42),
            threshold_metric=model_cfg.get("target_metric", "f2"),
            optimize_for=cfg.optimize_for,
            recall_min=cfg.recall_min,
        )
    _task.__name__ = f"task_train_{model_key}_{target_name}"
    return _task


def make_task_evaluate_model(target_name: str, model_key: str):
    def _task(**kwargs):
        import pandas as pd
        from src.utils.io import read_parquet, read_json, write_json
        from src.models.trainer import load_model
        from src.evaluation.metrics import compute_classification_metrics
        from src.evaluation.subgroups import cross_validate_model

        proc_dir = cfg.output_path() / "data_processed"
        splits_dir = proc_dir / target_name / "splits"
        models_dir = cfg.output_path() / "models" / target_name

        X_test = read_parquet(splits_dir / "X_test.parquet")
        y_test = read_parquet(splits_dir / "y_test.parquet")["target"]
        X_train = read_parquet(splits_dir / "X_train.parquet")
        y_train = read_parquet(splits_dir / "y_train.parquet")["target"]

        artifact = read_json(models_dir / f"{model_key}_metrics.json")
        threshold = artifact.get("Threshold", 0.5)

        model = load_model(models_dir / f"{model_key}_model.joblib")
        X_test_num = X_test.apply(pd.to_numeric, errors="coerce").fillna(-1)
        y_proba = model.predict_proba(X_test_num)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        test_metrics = compute_classification_metrics(y_test, y_pred, y_proba, threshold)
        cv_metrics = cross_validate_model(
            model, X_train, y_train,
            n_folds=cfg.cross_validation.get("n_folds", 10),
        )

        write_json({"test": test_metrics, "cv": cv_metrics},
                   models_dir / f"{model_key}_eval.json")
        logger.info(
            f"[{target_name}/{model_key}] Test F2={test_metrics['F2']:.3f} | "
            f"CV F2={cv_metrics['F2_mean']:.3f}±{cv_metrics['F2_std']:.3f}"
        )
    _task.__name__ = f"task_evaluate_{model_key}_{target_name}"
    return _task


def make_task_explainability(target_name: str, model_key: str):
    def _task(**kwargs):
        import pandas as pd
        from src.utils.io import read_parquet, read_json
        from src.models.trainer import load_model
        from src.evaluation.explainability import compute_global_explainability, build_case_review_table

        proc_dir = cfg.output_path() / "data_processed"
        splits_dir = proc_dir / target_name / "splits"
        models_dir = cfg.output_path() / "models" / target_name
        expl_dir = cfg.output_path() / "reports" / "explainability" / target_name
        expl_dir.mkdir(parents=True, exist_ok=True)

        X_test = read_parquet(splits_dir / "X_test.parquet")
        y_test = read_parquet(splits_dir / "y_test.parquet")["target"]

        artifact = read_json(models_dir / f"{model_key}_metrics.json")
        threshold = artifact.get("Threshold", 0.5)

        model = load_model(models_dir / f"{model_key}_model.joblib")
        X_test_num = X_test.apply(pd.to_numeric, errors="coerce").fillna(-1)
        y_proba = model.predict_proba(X_test_num)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        df_global = compute_global_explainability(model, X_test_num, y_test)
        df_global.to_csv(expl_dir / f"explainability_global_{model_key}.csv", index=False)

        top_features = df_global["Feature"].tolist()
        df_cases = build_case_review_table(X_test, y_test, y_pred, y_proba, top_features)
        df_cases.to_csv(expl_dir / f"explainability_cases_{model_key}.csv", index=False)

        logger.info(
            f"[{target_name}/{model_key}] Explicabilidad: {len(df_global)} features | "
            f"{len(df_cases)} casos auditables"
        )
    _task.__name__ = f"task_explainability_{model_key}_{target_name}"
    return _task


def make_task_model_plots(target_name: str, model_key: str):
    def _task(**kwargs):
        import pandas as pd
        from src.utils.io import read_parquet, read_json
        from src.models.trainer import load_model
        from src.evaluation.metrics import find_optimal_threshold
        from src.reports.model_plots import plot_roc_pr, plot_confusion_matrix, plot_threshold_curve

        proc_dir = cfg.output_path() / "data_processed"
        splits_dir = proc_dir / target_name / "splits"
        models_dir = cfg.output_path() / "models" / target_name
        plots_dir = cfg.output_path() / "plots" / target_name

        X_test = read_parquet(splits_dir / "X_test.parquet")
        y_test = read_parquet(splits_dir / "y_test.parquet")["target"]
        artifact = read_json(models_dir / f"{model_key}_metrics.json")
        threshold = artifact.get("Threshold", 0.5)

        model = load_model(models_dir / f"{model_key}_model.joblib")
        X_test_num = X_test.apply(pd.to_numeric, errors="coerce").fillna(-1)
        y_proba = model.predict_proba(X_test_num)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        _, _, thresholds, scores = find_optimal_threshold(y_test, y_proba, metric="f2")
        plot_roc_pr(y_test, y_proba, model_key, target_name, plots_dir)
        plot_confusion_matrix(y_test, y_pred, model_key, target_name, plots_dir)
        plot_threshold_curve(thresholds, scores, threshold, model_key, target_name, plots_dir)
    _task.__name__ = f"task_model_plots_{model_key}_{target_name}"
    return _task


def task_pre_post_analysis(**kwargs):
    from src.reports.pre_post_analysis import run_pre_post_linkage_analysis

    proc_dir = cfg.output_path() / "data_processed"
    output_dir = cfg.output_path() / "reports"
    run_pre_post_linkage_analysis(
        proc_dir=proc_dir,
        output_dir=output_dir,
        active_targets=ACTIVE_TARGETS,
        random_state=cfg.train_test_split.get("random_state", 42),
    )


def task_generate_comparison_report(**kwargs):
    from src.evaluation.comparison import aggregate_model_results
    from src.reports.comparison_plots import plot_comparison_heatmap, plot_ranking_bars

    models_dir = cfg.output_path() / "models"
    plots_dir = cfg.output_path() / "plots"
    reports_dir = cfg.output_path() / "reports"

    df = aggregate_model_results(models_dir, reports_dir)
    if not df.empty:
        plot_comparison_heatmap(df, "F2", plots_dir)
        plot_comparison_heatmap(df, "ROC_AUC", plots_dir)
        plot_ranking_bars(df, "F2", plots_dir)
        logger.info("Reporte comparativo generado")


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
    explainability_tasks: dict = {}
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
                python_callable=make_task_train_model(target, model),
            )
            evaluate_tasks[key] = PythonOperator(
                task_id=f"evaluate__{model}__{target}",
                python_callable=make_task_evaluate_model(target, model),
            )
            explainability_tasks[key] = PythonOperator(
                task_id=f"explainability__{model}__{target}",
                python_callable=make_task_explainability(target, model),
            )
            plot_tasks[key] = PythonOperator(
                task_id=f"model_plots__{model}__{target}",
                python_callable=make_task_model_plots(target, model),
            )

    pre_post_task = PythonOperator(
        task_id="pre_post_analysis",
        python_callable=task_pre_post_analysis,
    )
    comparison_report = PythonOperator(
        task_id="generate_comparison_report",
        python_callable=task_generate_comparison_report,
    )

    # ── Dependencias ─────────────────────────────────────────────────────────
    validate_raw >> eda_preop_raw >> clean_data >> eda_preop_clean

    for target in ACTIVE_TARGETS:
        eda_preop_clean >> extract_target_tasks[target]
        extract_target_tasks[target] >> merge_tasks[target]
        merge_tasks[target] >> eda_posop_tasks[target]
        merge_tasks[target] >> select_features_tasks[target]
        [eda_posop_tasks[target], select_features_tasks[target]] >> eda_correlation_tasks[target]

        select_features_tasks[target] >> pre_post_task

        for model in ENABLED_MODELS:
            key = f"{model}__{target}"
            eda_correlation_tasks[target] >> train_tasks[key]
            train_tasks[key] >> evaluate_tasks[key]
            evaluate_tasks[key] >> explainability_tasks[key]
            explainability_tasks[key] >> plot_tasks[key]
            plot_tasks[key] >> comparison_report

    pre_post_task >> comparison_report
