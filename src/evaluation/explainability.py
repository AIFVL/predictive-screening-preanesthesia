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
