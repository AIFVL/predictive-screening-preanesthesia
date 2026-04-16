# src/evaluation/explainability.py
"""
Explicabilidad global y revisión de casos.
Migrado fielmente desde utils/modeling.py y utils/multi_version/modeling_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.logger import get_logger

logger = get_logger("evaluation.explainability")


def compute_global_explainability(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Retorna importancia global de features para explicabilidad.
    Prioridad:
    1) feature_importances_ (árboles/boosting)
    2) coef_ (modelos lineales)
    3) permutation importance (fallback genérico, scoring='roc_auc', n_repeats=5)

    Retorna DataFrame con columnas: Feature, Importance, Source.
    """
    feature_names = list(X.columns)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        df_imp = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
                "Source": "native_feature_importance",
            }
        )
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        df_imp = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": np.abs(coef),
                "Source": "abs_coef",
            }
        )
    else:
        perm = permutation_importance(
            model,
            X,
            y,
            scoring="roc_auc",
            n_repeats=5,
            random_state=random_state,
            n_jobs=-1,
        )
        df_imp = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": perm.importances_mean,
                "Source": "permutation_importance",
            }
        )

    df_imp = df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp.head(top_n).copy()


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
    Clasifica cada caso como FN/FP/TP/TN y genera una columna 'reason' con los valores
    de los top-3 features más predictivos.

    FN/TP se ordenan por y_proba descendente; FP/TN por y_proba ascendente.
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


# ── SHAP ──────────────────────────────────────────────────────────────────────

#: Modelos con TreeExplainer nativo (exacto y rápido).
_TREE_MODELS = (
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "HistGradientBoostingClassifier",
    "GradientBoostingClassifier",
)

#: Modelos con LinearExplainer nativo.
_LINEAR_MODELS = (
    "LogisticRegression",
    "LinearSVC",
    "SGDClassifier",
    "Ridge",
)


def _unwrap_for_shap(model):
    """
    Devuelve (estimator_para_shap, es_unwrapped).
    Para CalibratedClassifierCV extrae el estimador base del primer fold,
    que es el que tiene feature_importances_ / coef_ y es compatible con
    TreeExplainer / LinearExplainer.
    """
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        calibrated = getattr(model, "calibrated_classifiers_", None)
        if calibrated:
            base = getattr(calibrated[0], "estimator", None)
            if base is not None:
                return base, True
    return model, False


def compute_shap_values(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str = "",
    background_sample_size: int = 200,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Calcula SHAP values para la clase positiva (índice 1).

    Estrategia por tipo de modelo:
    - Árbol (RF, ET, XGB, LGBM, HistGB): TreeExplainer — exacto, rápido.
    - Lineal (LR):                        LinearExplainer — exacto.
    - Ensamble (Stacking, Voting) / MLP:  KernelExplainer sobre muestra de
                                          X_background (≤ background_sample_size).

    Parámetros
    ----------
    model            : modelo entrenado (puede ser CalibratedClassifierCV).
    X_background     : datos de referencia (típicamente X_train) para el
                       background de KernelExplainer.
    X_explain        : datos a explicar (típicamente X_test o los FN).
    model_name       : nombre del modelo (str) para elegir el explainer.
    background_sample_size : máx filas del background para KernelExplainer.
    random_state     : semilla para el muestreo del background.

    Retorna
    -------
    shap_values      : array (n_samples, n_features) — contribución de cada
                       feature a la predicción de la clase positiva.
    expected_value   : valor base (probabilidad media del modelo sobre background).
    """
    import shap

    inner, was_unwrapped = _unwrap_for_shap(model)
    cls_name = type(inner).__name__

    # ── TreeExplainer ──────────────────────────────────────────────────────
    if cls_name in _TREE_MODELS:
        explainer = shap.TreeExplainer(
            inner,
            feature_perturbation="interventional",
            model_output="probability",
            data=X_background,
        )
        sv = explainer.shap_values(X_explain, check_additivity=False)
        # shap_values puede ser lista [neg, pos] (sklearn trees) o array (XGB/LGBM)
        if isinstance(sv, list):
            shap_vals = sv[1]
        else:
            shap_vals = sv
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[1])
        else:
            ev = float(ev)
        return shap_vals, ev

    # ── LinearExplainer ────────────────────────────────────────────────────
    if cls_name in _LINEAR_MODELS:
        explainer = shap.LinearExplainer(inner, X_background)
        sv = explainer.shap_values(X_explain)
        if isinstance(sv, list):
            shap_vals = sv[1]
        else:
            shap_vals = sv
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[1])
        else:
            ev = float(ev)
        return shap_vals, ev

    # ── KernelExplainer (fallback) ─────────────────────────────────────────
    # Para Stacking, Voting, MLP y cualquier modelo no reconocido.
    # Se muestrea el background para mantener tiempos razonables.
    logger.warning(
        f"[SHAP] {cls_name} no tiene explainer nativo — usando KernelExplainer "
        f"(más lento). Background sample = {background_sample_size} filas."
    )
    rng = np.random.default_rng(random_state)
    n = min(background_sample_size, len(X_background))
    idx = rng.choice(len(X_background), size=n, replace=False)
    bg_sample = X_background.iloc[idx]

    def _predict_proba_pos(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(_predict_proba_pos, bg_sample)
    shap_vals = explainer.shap_values(X_explain, silent=True)
    ev = float(explainer.expected_value)
    return shap_vals, ev
