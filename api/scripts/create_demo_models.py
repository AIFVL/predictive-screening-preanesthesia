"""
Creates 3 demo model artifacts in models/ for testing the API.
Run from project root: python -m api.scripts.create_demo_models
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

np.random.seed(42)

FEATURE_COLS = [
    "Edad", "Peso", "Talla", "IMC",
    "Tensión Arterial Sistólica (mm/Hg)", "Tensión Arterial Diastólica (mm/Hg)",
    "Frecuencia Cardiaca", "SpO2",
    "severity_score_high_severity_proc", "severity_score_critical_proc",
    "severity_ordinal_proc", "severity_ordinal_dx",
    "Prótesis Dental_movil", "Alérgeno_med_opioides",
    "ASA", "n_dx", "n_proc", "n_med",
    "flag_cancelacion_hist", "flag_via_aerea_hist",
]

FILL_ZERO = [
    "severity_score_high_severity_proc", "severity_score_critical_proc",
    "severity_ordinal_proc", "severity_ordinal_dx", "Prótesis Dental_movil",
    "Alérgeno_med_opioides", "n_dx", "n_proc", "n_med",
    "flag_cancelacion_hist", "flag_via_aerea_hist",
]
FILL_MEDIAN = [
    "Edad", "Peso", "Talla", "IMC",
    "Tensión Arterial Sistólica (mm/Hg)", "Tensión Arterial Diastólica (mm/Hg)",
    "Frecuencia Cardiaca", "SpO2", "ASA",
]


def make_dataset(n=600):
    X = pd.DataFrame({
        "Edad":                                   np.random.randint(18, 85, n).astype(float),
        "Peso":                                   np.random.normal(72, 15, n),
        "Talla":                                  np.random.normal(168, 10, n),
        "IMC":                                    np.random.normal(26, 5, n),
        "Tensión Arterial Sistólica (mm/Hg)":     np.random.normal(125, 20, n),
        "Tensión Arterial Diastólica (mm/Hg)":    np.random.normal(80, 12, n),
        "Frecuencia Cardiaca":                    np.random.normal(75, 15, n),
        "SpO2":                                   np.random.normal(97, 2, n),
        "severity_score_high_severity_proc":      np.random.randint(0, 3, n).astype(float),
        "severity_score_critical_proc":           np.random.randint(0, 2, n).astype(float),
        "severity_ordinal_proc":                  np.random.randint(0, 4, n).astype(float),
        "severity_ordinal_dx":                    np.random.randint(0, 4, n).astype(float),
        "Prótesis Dental_movil":                  np.random.randint(0, 2, n).astype(float),
        "Alérgeno_med_opioides":                  np.random.randint(0, 2, n).astype(float),
        "ASA":                                    np.random.randint(1, 5, n).astype(float),
        "n_dx":                                   np.random.randint(0, 10, n).astype(float),
        "n_proc":                                 np.random.randint(1, 5, n).astype(float),
        "n_med":                                  np.random.randint(0, 15, n).astype(float),
        "flag_cancelacion_hist":                  np.random.randint(0, 2, n).astype(float),
        "flag_via_aerea_hist":                    np.random.randint(0, 2, n).astype(float),
    })
    y = (
        (X["ASA"] >= 3).astype(int)
        | (X["severity_score_critical_proc"] >= 1).astype(int)
        | (X["flag_via_aerea_hist"] == 1).astype(int)
        | ((X["Edad"] > 70) & (X["n_dx"] > 5)).astype(int)
    ).clip(0, 1)
    return X, y


def build_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "ROC-AUC":   round(float(roc_auc_score(y_true, y_proba)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Threshold": threshold,
    }


def importance_list(model, cols):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return []
    return sorted(
        [{"feature": c, "importance": float(v)} for c, v in zip(cols, vals)],
        key=lambda x: x["importance"],
        reverse=True,
    )


def save_artifact(path, model, preprocessor, threshold, metrics, version, model_name, model_type, X_train):
    artifact = {
        "model":                  model,
        "preprocessor":           preprocessor,
        "feature_columns":        FEATURE_COLS,
        "optimal_threshold":      threshold,
        "training_medians":       {c: float(X_train[c].median()) for c in FEATURE_COLS},
        "imputation_strategies":  {"fill_zero": FILL_ZERO, "fill_median": FILL_MEDIAN},
        "version":                version,
        "model_name":             model_name,
        "model_type":             model_type,
        "metrics":                metrics,
        "feature_importance":     importance_list(model, FEATURE_COLS),
        "created_at":             "2026-03-11",
        "description":            f"{model_name} — {version} (demo)",
        "min_age":                18,
        "target_description":     "1 = paciente requiere evaluación preanestésica, 0 = no indicada",
    }
    joblib.dump(artifact, path)
    print(f"  Saved: {path}  |  ROC-AUC={metrics['ROC-AUC']}  Recall={metrics['Recall']}")


def main():
    Path("models").mkdir(exist_ok=True)
    X, y = make_dataset()
    print(f"Dataset: {len(X)} rows, target prevalence={y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Tree preprocessor (shared by HGB + RF) ────────────────────────────────
    prep_tree = ColumnTransformer(
        [
            ("zero",   SimpleImputer(strategy="constant", fill_value=0), FILL_ZERO),
            ("median", SimpleImputer(strategy="median"),                  FILL_MEDIAN),
        ],
        remainder="drop",
    )
    Xtr = pd.DataFrame(prep_tree.fit_transform(X_train), columns=FILL_ZERO + FILL_MEDIAN)[FEATURE_COLS]
    Xte = pd.DataFrame(prep_tree.transform(X_test),      columns=FILL_ZERO + FILL_MEDIAN)[FEATURE_COLS]

    # ── 1. HistGradientBoosting / target_b ────────────────────────────────────
    print("Training HistGradientBoosting (target_b_clinicamente_relevante) ...")
    hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42).fit(Xtr, y_train)
    proba = hgb.predict_proba(Xte)[:, 1]
    save_artifact(
        "models/histgb_target_b.joblib", hgb, prep_tree, 0.38,
        build_metrics(y_test, proba, 0.38),
        "target_b_clinicamente_relevante", "HistGradientBoosting", "tree", X_train,
    )

    # ── 2. Random Forest / target_a ───────────────────────────────────────────
    print("Training RandomForest (target_a_sensible) ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42).fit(Xtr, y_train)
    proba = rf.predict_proba(Xte)[:, 1]
    save_artifact(
        "models/rf_target_a.joblib", rf, prep_tree, 0.35,
        build_metrics(y_test, proba, 0.35),
        "target_a_sensible", "Random Forest", "tree", X_train,
    )

    # ── 3. Logistic Regression / target_c ─────────────────────────────────────
    print("Training Logistic Regression (target_c_alta_severidad) ...")
    prep_lin = Pipeline([
        ("imputer", ColumnTransformer(
            [
                ("zero",   SimpleImputer(strategy="constant", fill_value=0), FILL_ZERO),
                ("median", SimpleImputer(strategy="median"),                  FILL_MEDIAN),
            ],
            remainder="drop",
        )),
        ("scaler", RobustScaler()),
    ])
    Xtr_lin = prep_lin.fit_transform(X_train)
    Xte_lin = prep_lin.transform(X_test)
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42).fit(Xtr_lin, y_train)
    proba = lr.predict_proba(Xte_lin)[:, 1]
    save_artifact(
        "models/lr_target_c.joblib", lr, prep_lin, 0.42,
        build_metrics(y_test, proba, 0.42),
        "target_c_alta_severidad", "Regresión Logística", "linear", X_train,
    )

    print("\nDemo models ready in models/")


if __name__ == "__main__":
    main()
