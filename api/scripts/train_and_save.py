"""
Train and Save Model Artifact
==============================
Trains a model using the existing pipeline and saves a .joblib artifact in
the format expected by the Preanesthesia Screening API.

Usage
-----
    python -m api.scripts.train_and_save \\
        --data      OPERA_COMPLETO.xlsx \\
        --features  variables_seleccionadas.csv \\
        --version   target_b_clinicamente_relevante \\
        --model     HistGradientBoosting \\
        --output    models/histgb_target_b.joblib \\
        --optimize          # run Optuna hyperparameter search (30 trials)

The script integrates with utils/ from the project root. Run from the
repository root so that the utils package is importable.
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Make utils importable from project root ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_engineering import (
    ENCODING_FIX_MAP,
    get_imputation_strategies,
    resolve_feature_columns,
    sanitize_features_for_subset,
)
from utils.modeling import (
    compute_classification_metrics,
    evaluate_model_performance,
    find_optimal_threshold,
    get_models_definitions,
    instantiate_model_from_params,
    optimize_best_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    data_path: Path,
    features_path: Path | None,
    target_col: str = "target",
    min_age: int = 18,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load dataset, resolve features, and return X, y, feature_columns."""
    logger.info("Loading dataset: %s", data_path)
    df = pd.read_excel(data_path) if data_path.suffix == ".xlsx" else pd.read_csv(data_path)

    df = df.rename(columns=ENCODING_FIX_MAP)

    if "Edad" in df.columns:
        df = df[df["Edad"] >= min_age].copy()
        logger.info("After adult filter (Edad >= %d): %d rows", min_age, len(df))

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available: {list(df.columns)[:10]}…"
        )

    if features_path is not None and features_path.exists():
        features_meta = pd.read_csv(features_path)
        selected_names = features_meta["Variable"].tolist()
        feature_columns, missing = resolve_feature_columns(selected_names, df.columns.tolist(), features_meta)
        if missing:
            logger.warning("Could not resolve %d features: %s…", len(missing), missing[:5])
    else:
        feature_columns = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
        logger.info("No features file supplied – using all %d numeric columns.", len(feature_columns))

    feature_columns, dropped = sanitize_features_for_subset(df, feature_columns)
    if dropped:
        logger.info("Dropped %d constant columns: %s…", len(dropped), dropped[:5])

    X = df[feature_columns].copy()
    y = df[target_col].astype(int)

    pos_rate = y.mean()
    logger.info(
        "Dataset: %d rows, %d features, target prevalence=%.1f%%",
        len(X), len(feature_columns), pos_rate * 100,
    )
    return X, y, feature_columns


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor(
    feature_columns: list[str],
    strategies: dict[str, list[str]],
    model_type: str = "tree",
) -> ColumnTransformer:
    """
    Build a fitted-ready ColumnTransformer that imputes missing values.
    For linear models, StandardScaler is added after imputation.
    """
    fill_zero_cols = [c for c in strategies["fill_zero"] if c in feature_columns]
    fill_median_cols = [c for c in strategies["fill_median"] if c in feature_columns]
    remaining = [c for c in feature_columns if c not in fill_zero_cols and c not in fill_median_cols]

    transformers = []
    if fill_zero_cols:
        transformers.append(
            ("zero_impute", SimpleImputer(strategy="constant", fill_value=0), fill_zero_cols)
        )
    if fill_median_cols:
        transformers.append(
            ("median_impute", SimpleImputer(strategy="median"), fill_median_cols)
        )
    if remaining:
        transformers.append(
            ("fallback_impute", SimpleImputer(strategy="constant", fill_value=0), remaining)
        )

    if model_type == "linear":
        from sklearn.preprocessing import RobustScaler
        steps = [
            ("imputer", ColumnTransformer(transformers, remainder="drop")),
            ("scaler", RobustScaler()),
        ]
        return Pipeline(steps)

    return ColumnTransformer(transformers, remainder="drop")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    optimize: bool,
    random_state: int,
    scale_pos_weight: float,
    n_trials: int = 30,
):
    """Train (and optionally optimise) the requested model. Returns fitted model + metrics."""
    models = get_models_definitions(random_state=random_state, scale_pos_weight=scale_pos_weight)

    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(models.keys())}"
        )

    if optimize:
        logger.info("Running Optuna hyperparameter search (%d trials) …", n_trials)
        study = optimize_best_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            n_trials=n_trials,
            metric="roc_auc",
        )
        logger.info(
            "Best params: %s  (val ROC-AUC=%.4f)",
            study.best_params,
            study.best_value,
        )
        model = instantiate_model_from_params(
            model_name=model_name,
            best_params=study.best_params,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train)
    else:
        model = models[model_name]
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, best_f2, _, _ = find_optimal_threshold(
        y_test, y_proba, metric="f2", recall_min=0.75
    )
    logger.info("Optimal threshold (F2): %.3f  (F2=%.4f)", optimal_threshold, best_f2)

    y_pred = (y_proba >= optimal_threshold).astype(int)
    metrics = compute_classification_metrics(y_test, y_pred, y_proba, optimal_threshold)
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    return model, metrics, float(optimal_threshold)


# ── Artefact assembly ─────────────────────────────────────────────────────────

def build_artifact(
    model,
    preprocessor,
    X_train: pd.DataFrame,
    feature_columns: list[str],
    strategies: dict[str, list[str]],
    optimal_threshold: float,
    metrics: dict,
    model_name: str,
    model_type: str,
    version: str,
    description: str = "",
    min_age: int = 18,
) -> dict:
    """Assemble the artifact dict consumed by the API."""
    training_medians = {
        col: float(X_train[col].median())
        for col in feature_columns
        if col in X_train.columns
    }

    feature_importance = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance = sorted(
            [{"feature": col, "importance": float(imp)} for col, imp in zip(feature_columns, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        feature_importance = sorted(
            [{"feature": col, "importance": float(imp)} for col, imp in zip(feature_columns, coef)],
            key=lambda x: x["importance"],
            reverse=True,
        )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "optimal_threshold": optimal_threshold,
        "training_medians": training_medians,
        "imputation_strategies": strategies,
        "version": version,
        "model_name": model_name,
        "model_type": model_type,
        "metrics": metrics,
        "feature_importance": feature_importance[:50],
        "created_at": date.today().isoformat(),
        "description": description or f"{model_name} trained on {version}",
        "min_age": min_age,
        "target_description": (
            "1 = patient requires preanesthesia evaluation, "
            "0 = evaluation not indicated"
        ),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model and save a .joblib artifact for the Preanesthesia API."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to OPERA_COMPLETO*.xlsx or CSV")
    parser.add_argument("--features", type=Path, default=None, help="Path to variables_seleccionadas.csv")
    parser.add_argument("--target-col", default="target", help="Target column name (default: 'target')")
    parser.add_argument(
        "--version",
        default="target_b_clinicamente_relevante",
        help="Clinical target version name (informational, stored in artifact)",
    )
    parser.add_argument(
        "--model",
        default="HistGradientBoosting",
        help="Model name from get_models_definitions() (default: HistGradientBoosting)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output path for .joblib artifact")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (default: 30)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=18, help="Minimum patient age (default: 18)")
    parser.add_argument("--description", default="", help="Human-readable model description")
    return parser.parse_args()


def main():
    args = parse_args()

    from sklearn.model_selection import StratifiedShuffleSplit

    X, y, feature_columns = load_data(
        data_path=args.data,
        features_path=args.features,
        target_col=args.target_col,
        min_age=args.min_age,
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    strategies = get_imputation_strategies(feature_columns)
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    logger.info("scale_pos_weight=%.2f  (positive rate=%.1f%%)", scale_pos_weight, pos_rate * 100)

    model_name = args.model
    tree_models = {"Random Forest", "XGBoost", "HistGradientBoosting", "Extra Trees",
                   "Gradient Boosting", "Árbol de Decisión"}
    linear_models = {"Regresión Logística"}
    model_type = (
        "tree" if model_name in tree_models
        else "linear" if model_name in linear_models
        else "ensemble"
    )

    preprocessor = build_preprocessor(feature_columns, strategies, model_type=model_type)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_train_proc = pd.DataFrame(X_train_proc, columns=feature_columns)
    X_test_proc = pd.DataFrame(X_test_proc, columns=feature_columns)

    model, metrics, optimal_threshold = train(
        X_train=X_train_proc,
        y_train=y_train.reset_index(drop=True),
        X_test=X_test_proc,
        y_test=y_test.reset_index(drop=True),
        model_name=model_name,
        optimize=args.optimize,
        random_state=args.random_state,
        scale_pos_weight=scale_pos_weight,
        n_trials=args.n_trials,
    )

    artifact = build_artifact(
        model=model,
        preprocessor=preprocessor,
        X_train=X_train,
        feature_columns=feature_columns,
        strategies=strategies,
        optimal_threshold=optimal_threshold,
        metrics=metrics,
        model_name=model_name,
        model_type=model_type,
        version=args.version,
        description=args.description,
        min_age=args.min_age,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    logger.info("Artifact saved to '%s'.", args.output)
    print("\n" + "=" * 60)
    print(f"  Model: {model_name}  |  Version: {args.version}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Threshold: {optimal_threshold:.3f}")
    print(f"  ROC-AUC:  {metrics.get('ROC-AUC', 'n/a'):.4f}")
    print(f"  Recall:   {metrics.get('Recall', 'n/a'):.4f}")
    print(f"  F2:       {metrics.get('F2', 'n/a'):.4f}")
    print(f"  Artifact: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
