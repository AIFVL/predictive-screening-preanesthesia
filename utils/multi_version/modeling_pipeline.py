from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.feature_engineering import ENCODING_FIX_MAP, sanitize_features_for_subset
import utils.modeling as mod


def extract_version_from_merged(path: Path) -> str:
    return path.stem.replace("OPERA_COMPLETO_", "")


def extract_version_from_features(path: Path) -> str:
    return path.stem.replace("_selected_features_list", "")


def load_features_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def prepare_xy_for_version(df: pd.DataFrame, selected_features: list[str]):
    df_local = df.copy()

    if "Edad" in df_local.columns:
        df_local = df_local[df_local["Edad"] >= 18].reset_index(drop=True)

    df_local = df_local.rename(columns=ENCODING_FIX_MAP)
    selected_features = [ENCODING_FIX_MAP.get(c, c) for c in selected_features]
    selected_features = [c for c in selected_features if c in df_local.columns]

    selected_features, _ = sanitize_features_for_subset(df_local, selected_features)
    if not selected_features:
        raise ValueError("No quedaron features válidas después de sanitización")

    X = df_local[selected_features].copy()
    y = df_local["target"].astype(int).copy()
    return X, y, selected_features


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split estratificado en train/val/test.
    val_size y test_size son proporciones del total.
    """
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size y test_size deben ser >0 y su suma <1")

    # 1) Separar test del total
    splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_test.split(X, y))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # 2) Separar validation del subconjunto train_val
    relative_val_size = val_size / (1 - test_size)
    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx_rel, val_idx_rel = next(splitter_val.split(X_train_val, y_train_val))

    X_train = X_train_val.iloc[train_idx_rel]
    y_train = y_train_val.iloc[train_idx_rel]
    X_val = X_train_val.iloc[val_idx_rel]
    y_val = y_train_val.iloc[val_idx_rel]

    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_binary_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balancea un dataset binario vía undersampling de la clase mayoritaria.
    """
    classes = sorted(y.unique().tolist())
    if len(classes) != 2:
        return X, y

    idx_0 = y[y == classes[0]].index
    idx_1 = y[y == classes[1]].index
    n_min = min(len(idx_0), len(idx_1))
    if n_min == 0:
        return X, y

    rng = np.random.RandomState(random_state)
    sample_0 = rng.choice(idx_0, size=n_min, replace=False)
    sample_1 = rng.choice(idx_1, size=n_min, replace=False)
    balanced_idx = np.concatenate([sample_0, sample_1])
    rng.shuffle(balanced_idx)

    return X.loc[balanced_idx].copy(), y.loc[balanced_idx].copy()


def build_case_review_table(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    top_features: list[str],
    max_cases_per_group: int = 10,
) -> pd.DataFrame:
    """
    Construye tabla de casos para auditoría clínica con motivos basados en top features.
    """
    eval_df = pd.DataFrame(
        {
            "case_index": y_test.index,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )

    eval_df["case_type"] = "OTHER"
    eval_df.loc[(eval_df["y_true"] == 1) & (eval_df["y_pred"] == 1), "case_type"] = "TP"
    eval_df.loc[(eval_df["y_true"] == 0) & (eval_df["y_pred"] == 0), "case_type"] = "TN"
    eval_df.loc[(eval_df["y_true"] == 1) & (eval_df["y_pred"] == 0), "case_type"] = "FN"
    eval_df.loc[(eval_df["y_true"] == 0) & (eval_df["y_pred"] == 1), "case_type"] = "FP"

    selected_blocks = []
    for case_type in ["FN", "FP", "TP", "TN"]:
        block = eval_df[eval_df["case_type"] == case_type].copy()
        if block.empty:
            continue

        if case_type in ["FN", "TP"]:
            block = block.sort_values("y_proba", ascending=False)
        else:
            block = block.sort_values("y_proba", ascending=True)

        selected_blocks.append(block.head(max_cases_per_group))

    if not selected_blocks:
        return pd.DataFrame(columns=["case_index", "case_type", "y_true", "y_pred", "y_proba", "reason"])

    selected = pd.concat(selected_blocks, ignore_index=True)

    def reason_from_top_features(case_idx):
        if case_idx not in X_test.index:
            return ""
        values = []
        for feature in top_features[:3]:
            if feature in X_test.columns:
                v = X_test.loc[case_idx, feature]
                values.append(f"{feature}={round(float(v), 4) if pd.notnull(v) else 'NA'}")
        return " | ".join(values)

    selected["reason"] = selected["case_index"].apply(reason_from_top_features)
    return selected[["case_index", "case_type", "y_true", "y_pred", "y_proba", "reason"]].copy()


def run_modeling_pipeline(
    merged_dir: Path,
    features_versioned_dir: Path,
    output_dir: Path,
    random_state: int = 42,
    active_versions: list[str] | None = None,
    balance_mode: str = "none",
    val_size: float = 0.2,
    test_size: float = 0.2,
    threshold_metric: str = "balanced_accuracy",
    threshold_prevalence_tolerance: float = 0.5,
    threshold_restrict_prevalence: bool = False,
    optimize_for: str | None = None,
    recall_min: float = 0.80,
):
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    feature_files = sorted(features_versioned_dir.glob("*_selected_features_list.txt"))

    if not merged_files:
        raise FileNotFoundError("No se encontraron datasets mergeados en data/merged_versions")
    if not feature_files:
        raise FileNotFoundError("No se encontraron listas de features en features/versioned")

    merged_map = {extract_version_from_merged(path): path for path in merged_files}
    features_map = {extract_version_from_features(path): path for path in feature_files}

    common_versions = sorted(set(merged_map).intersection(features_map))
    ignored_versions = []

    if active_versions is not None:
        active_set = set(active_versions)
        ignored_versions = [version for version in common_versions if version not in active_set]
        common_versions = [version for version in common_versions if version in active_set]

    if not common_versions:
        raise ValueError("No hay versiones en común entre datasets mergeados y listas de features")

    experiment_registry = pd.DataFrame(
        [
            {
                "version": v,
                "dataset_path": str(merged_map[v]),
                "features_path": str(features_map[v]),
            }
            for v in common_versions
        ]
    )

    if balance_mode not in {"none", "train_only", "train_test"}:
        raise ValueError("balance_mode debe ser 'none', 'train_only' o 'train_test'")

    if threshold_metric not in {"f1", "f2", "balanced_accuracy"}:
        raise ValueError("threshold_metric debe ser 'f1', 'f2' o 'balanced_accuracy'")
    if threshold_prevalence_tolerance < 0:
        raise ValueError("threshold_prevalence_tolerance debe ser >= 0")

    results_rows = []
    predictions_by_version = {}
    y_test_by_version = {}
    y_val_by_version = {}
    X_test_by_version = {}

    for version in common_versions:
        df_version = pd.read_excel(merged_map[version])
        selected_features = load_features_list(features_map[version])

        X, y, selected_features = prepare_xy_for_version(df_version, selected_features)
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
            X,
            y,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )

        train_pos_rate_original = float(y_train.mean()) if len(y_train) else np.nan
        val_pos_rate_original = float(y_val.mean()) if len(y_val) else np.nan
        test_pos_rate_original = float(y_test.mean()) if len(y_test) else np.nan

        if balance_mode == "train_only":
            X_train, y_train = balance_binary_dataset(X_train, y_train, random_state=random_state)

        train_pos_rate_used = float(y_train.mean()) if len(y_train) else np.nan
        val_pos_rate_used = float(y_val.mean()) if len(y_val) else np.nan
        test_pos_rate_used = float(y_test.mean()) if len(y_test) else np.nan

        pos_weight_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models_dict = mod.get_models_definitions(
            random_state=random_state,
            scale_pos_weight=pos_weight_ratio,
        )

        predictions_by_version[version] = {}
        y_test_by_version[version] = y_test
        y_val_by_version[version] = y_val
        X_test_by_version[version] = X_test

        for model_name, model in models_dict.items():
            model.fit(X_train, y_train)

            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_prevalence = float(np.mean(y_val))

            if threshold_restrict_prevalence:
                min_pos_rate = max(0.05, val_prevalence * (1 - threshold_prevalence_tolerance))
                max_pos_rate = min(0.95, val_prevalence * (1 + threshold_prevalence_tolerance))
            else:
                min_pos_rate = None
                max_pos_rate = None

            optimal_threshold, best_threshold_score, thresholds, threshold_scores = mod.find_optimal_threshold(
                y_val,
                y_val_proba,
                metric=threshold_metric,
                min_predicted_positive_rate=min_pos_rate,
                max_predicted_positive_rate=max_pos_rate,
                threshold_restrict_prevalence=threshold_restrict_prevalence,
                optimize_for=optimize_for,
                recall_min=recall_min,
            )

            y_test_proba = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

            predicted_positive_rate_val = float(np.mean((y_val_proba >= optimal_threshold).astype(int)))
            predicted_positive_rate_test = float(np.mean(y_test_pred))

            metrics = mod.compute_classification_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                y_proba=y_test_proba,
                threshold=optimal_threshold,
            )
            metrics["Threshold_Objective_Metric"] = threshold_metric
            metrics["Threshold_Objective_Score"] = float(best_threshold_score)
            val_f2 = mod.compute_classification_metrics(
                y_true=y_val,
                y_pred=(y_val_proba >= optimal_threshold).astype(int),
                y_proba=y_val_proba,
                threshold=optimal_threshold,
            )["F2"]
            metrics["F2_Validation_OptThreshold"] = float(val_f2)

            predictions_by_version[version][model_name] = {
                "y_pred": y_test_pred,
                "y_proba": y_test_proba,
                "y_val_proba": y_val_proba,
                "model": model,
            }

            row = {
                "Version": version,
                "Model": model_name,
                "N_Train": len(X_train),
                "N_Val": len(X_val),
                "N_Test": len(X_test),
                "N_Features": len(selected_features),
                "Balance_Mode": balance_mode,
                "Real_Prevalence": float(y.mean()),
                "Train_Prevalence_Used": train_pos_rate_used,
                "Train_Pos_Rate_Original": train_pos_rate_original,
                "Val_Pos_Rate_Original": val_pos_rate_original,
                "Test_Pos_Rate_Original": test_pos_rate_original,
                "Train_Pos_Rate_Used": train_pos_rate_used,
                "Val_Pos_Rate_Used": val_pos_rate_used,
                "Test_Pos_Rate_Used": test_pos_rate_used,
                "Predicted_Positive_Rate_Val": predicted_positive_rate_val,
                "Predicted_Positive_Rate_Test": predicted_positive_rate_test,
                "Threshold_Restrict_Prevalence": threshold_restrict_prevalence,
                "Threshold_Min_Pos_Rate": min_pos_rate,
                "Threshold_Max_Pos_Rate": max_pos_rate,
                "Optimize_For": optimize_for or threshold_metric,
            }
            row.update(metrics)
            results_rows.append(row)

    df_results = pd.DataFrame(results_rows).sort_values(
        ["ROC-AUC", "F2", "Accuracy", "Recall"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary_by_version = (
        df_results.sort_values(["Version", "ROC-AUC", "F2", "Accuracy"], ascending=[True, False, False, False])
        .groupby("Version", as_index=False)
        .first()
    )

    pivot_auc = df_results.pivot(index="Version", columns="Model", values="ROC-AUC")
    pivot_f2 = df_results.pivot(index="Version", columns="Model", values="F2")
    pivot_acc = df_results.pivot(index="Version", columns="Model", values="Accuracy")

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "baseline_results_by_version.csv"
    summary_path = output_dir / "best_model_by_version.csv"
    pivot_auc_path = output_dir / "pivot_auc.csv"
    pivot_f2_path = output_dir / "pivot_f2.csv"
    pivot_acc_path = output_dir / "pivot_accuracy.csv"

    df_results.to_csv(results_path, index=False)
    summary_by_version.to_csv(summary_path, index=False)
    pivot_auc.to_csv(pivot_auc_path, index=True)
    pivot_f2.to_csv(pivot_f2_path, index=True)
    pivot_acc.to_csv(pivot_acc_path, index=True)

    best_row = df_results.iloc[0]
    best_version = best_row["Version"]
    best_model = best_row["Model"]

    explainability_global_by_version = {}
    explainability_cases_by_version = {}
    explainability_registry = []

    for version in common_versions:
        version_rows = df_results[df_results["Version"] == version].sort_values(
            ["ROC-AUC", "F2", "Accuracy"],
            ascending=[False, False, False],
        )
        if version_rows.empty:
            continue

        best_model_version = version_rows.iloc[0]["Model"]
        model_obj = predictions_by_version[version][best_model_version]["model"]
        X_te = X_test_by_version[version]
        y_te = y_test_by_version[version]
        y_pr = predictions_by_version[version][best_model_version]["y_pred"]
        y_pb = predictions_by_version[version][best_model_version]["y_proba"]

        global_imp = mod.compute_global_explainability(
            model_obj,
            X_te,
            y_te,
            random_state=random_state,
            top_n=25,
        )
        top_features = global_imp["Feature"].tolist()

        cases = build_case_review_table(
            X_test=X_te,
            y_test=y_te,
            y_pred=y_pr,
            y_proba=y_pb,
            top_features=top_features,
            max_cases_per_group=10,
        )

        explainability_global_by_version[version] = global_imp
        explainability_cases_by_version[version] = cases

        global_path = output_dir / f"explainability_global_{version}.csv"
        cases_path = output_dir / f"explainability_cases_{version}.csv"

        global_imp.to_csv(global_path, index=False)
        cases.to_csv(cases_path, index=False)

        explainability_registry.append(
            {
                "Version": version,
                "Best_Model_For_Explainability": best_model_version,
                "Global_Explainability_Path": str(global_path),
                "Case_Review_Path": str(cases_path),
            }
        )

    df_explainability_registry = pd.DataFrame(explainability_registry).sort_values("Version").reset_index(drop=True)
    explainability_registry_path = output_dir / "explainability_registry.csv"
    df_explainability_registry.to_csv(explainability_registry_path, index=False)

    return {
        "experiment_registry": experiment_registry,
        "ignored_versions": sorted(set(ignored_versions)),
        "results": df_results,
        "summary_by_version": summary_by_version,
        "pivot_auc": pivot_auc,
        "pivot_f2": pivot_f2,
        "pivot_accuracy": pivot_acc,
        "predictions_by_version": predictions_by_version,
        "y_test_by_version": y_test_by_version,
        "y_val_by_version": y_val_by_version,
        "x_test_by_version": X_test_by_version,
        "explainability_global_by_version": explainability_global_by_version,
        "explainability_cases_by_version": explainability_cases_by_version,
        "explainability_registry": df_explainability_registry,
        "best_version": best_version,
        "best_model": best_model,
        "output_paths": {
            "results": results_path,
            "summary": summary_path,
            "pivot_auc": pivot_auc_path,
            "pivot_f2": pivot_f2_path,
            "pivot_accuracy": pivot_acc_path,
            "explainability_registry": explainability_registry_path,
        },
    }


def run_hyperparameter_experiments(
    merged_dir: Path,
    features_versioned_dir: Path,
    output_dir: Path,
    random_state: int = 42,
    active_versions: list[str] | None = None,
    balance_mode: str = "none",
    val_size: float = 0.2,
    test_size: float = 0.2,
    threshold_metric: str = "balanced_accuracy",
    threshold_prevalence_tolerance: float = 0.5,
    threshold_restrict_prevalence: bool = False,
    optimize_for: str | None = None,
    recall_min: float = 0.80,
    tune_models: list[str] | None = None,
    optimize_metric: str = "f2",
    n_trials: int = 40,
    feature_subset_fractions: tuple[float, ...] = (1.0,),
):
    """
    Ejecuta experimentos de tuning por versión y por combinaciones de subsets de features.
    """
    if balance_mode not in {"none", "train_only"}:
        raise ValueError("balance_mode debe ser 'none' o 'train_only'")
    if threshold_metric not in {"f1", "f2", "balanced_accuracy"}:
        raise ValueError("threshold_metric debe ser 'f1', 'f2' o 'balanced_accuracy'")
    if optimize_metric not in {"roc_auc", "f2"}:
        raise ValueError("optimize_metric debe ser 'roc_auc' o 'f2'")

    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    feature_files = sorted(features_versioned_dir.glob("*_selected_features_list.txt"))

    if not merged_files:
        raise FileNotFoundError("No se encontraron datasets mergeados en data/merged_versions")
    if not feature_files:
        raise FileNotFoundError("No se encontraron listas de features en features/versioned")

    merged_map = {extract_version_from_merged(path): path for path in merged_files}
    features_map = {extract_version_from_features(path): path for path in feature_files}

    common_versions = sorted(set(merged_map).intersection(features_map))
    ignored_versions = []

    if active_versions is not None:
        active_set = set(active_versions)
        ignored_versions = [version for version in common_versions if version not in active_set]
        common_versions = [version for version in common_versions if version in active_set]

    if not common_versions:
        raise ValueError("No hay versiones en común entre datasets mergeados y listas de features")

    if tune_models is None:
        tune_models = ["XGBoost", "Random Forest", "HistGradientBoosting", "Regresión Logística"]

    experiment_rows = []

    for version in common_versions:
        df_version = pd.read_excel(merged_map[version])
        selected_features = load_features_list(features_map[version])
        X_all, y_all, selected_features = prepare_xy_for_version(df_version, selected_features)

        subset_defs = []
        for frac in feature_subset_fractions:
            frac = float(frac)
            if frac <= 0 or frac > 1:
                continue
            n_features_subset = max(10, int(round(len(selected_features) * frac)))
            n_features_subset = min(n_features_subset, len(selected_features))
            subset_defs.append((frac, selected_features[:n_features_subset]))

        for frac, subset_features in subset_defs:
            subset_label = f"top_{int(frac * 100)}pct"
            X_subset = X_all[subset_features].copy()

            X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
                X_subset,
                y_all,
                val_size=val_size,
                test_size=test_size,
                random_state=random_state,
            )

            if balance_mode == "train_only":
                X_train, y_train = balance_binary_dataset(X_train, y_train, random_state=random_state)

            pos_weight_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

            for model_name in tune_models:
                study = mod.optimize_best_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    random_state=random_state,
                    scale_pos_weight=pos_weight_ratio,
                    n_trials=n_trials,
                    metric=optimize_metric,
                )

                tuned_model = mod.instantiate_model_from_params(
                    model_name=model_name,
                    best_params=study.best_params,
                    random_state=random_state,
                    scale_pos_weight=pos_weight_ratio,
                )
                tuned_model.fit(X_train, y_train)

                y_val_proba = tuned_model.predict_proba(X_val)[:, 1]
                val_prevalence = float(np.mean(y_val))

                if threshold_restrict_prevalence:
                    min_pos_rate = max(0.05, val_prevalence * (1 - threshold_prevalence_tolerance))
                    max_pos_rate = min(0.95, val_prevalence * (1 + threshold_prevalence_tolerance))
                else:
                    min_pos_rate = None
                    max_pos_rate = None

                optimal_threshold, best_threshold_score, _, _ = mod.find_optimal_threshold(
                    y_val,
                    y_val_proba,
                    metric=threshold_metric,
                    min_predicted_positive_rate=min_pos_rate,
                    max_predicted_positive_rate=max_pos_rate,
                    threshold_restrict_prevalence=threshold_restrict_prevalence,
                    optimize_for=optimize_for,
                    recall_min=recall_min,
                )

                y_test_proba = tuned_model.predict_proba(X_test)[:, 1]
                y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

                metrics_test = mod.compute_classification_metrics(
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_proba=y_test_proba,
                    threshold=optimal_threshold,
                )
                metrics_val = mod.compute_classification_metrics(
                    y_true=y_val,
                    y_pred=(y_val_proba >= optimal_threshold).astype(int),
                    y_proba=y_val_proba,
                    threshold=optimal_threshold,
                )

                row = {
                    "Version": version,
                    "Model": model_name,
                    "Feature_Subset": subset_label,
                    "N_Features": len(subset_features),
                    "N_Train": len(X_train),
                    "N_Val": len(X_val),
                    "N_Test": len(X_test),
                    "Balance_Mode": balance_mode,
                    "Threshold_Objective_Metric": threshold_metric,
                    "Threshold_Objective_Score": float(best_threshold_score),
                    "Tuning_Objective_Metric": optimize_metric,
                    "Tuning_Best_CV_Score": float(study.best_value),
                    "Tuning_N_Trials": int(n_trials),
                    "Best_Params": json.dumps(study.best_params, ensure_ascii=False),
                    "Validation_F2": float(metrics_val["F2"]),
                    "Validation_ROC-AUC": float(metrics_val["ROC-AUC"]),
                }
                row.update({f"Test_{k}": v for k, v in metrics_test.items()})
                experiment_rows.append(row)

    df_experiments = pd.DataFrame(experiment_rows)
    if df_experiments.empty:
        raise ValueError("No se generaron resultados de tuning")

    df_experiments = df_experiments.sort_values(
        ["Test_ROC-AUC", "Test_F2", "Test_Accuracy", "Validation_ROC-AUC"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_by_version = (
        df_experiments.sort_values(["Version", "Test_ROC-AUC", "Test_F2"], ascending=[True, False, False])
        .groupby("Version", as_index=False)
        .first()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = output_dir / "hyperparameter_experiments.csv"
    best_path = output_dir / "hyperparameter_best_by_version.csv"

    df_experiments.to_csv(experiments_path, index=False)
    best_by_version.to_csv(best_path, index=False)

    return {
        "experiments": df_experiments,
        "best_by_version": best_by_version,
        "ignored_versions": sorted(set(ignored_versions)),
        "output_paths": {
            "experiments": experiments_path,
            "best_by_version": best_path,
        },
    }
