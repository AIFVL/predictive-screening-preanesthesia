from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from src.features.engineering import apply_encoding_fixes, infer_candidate_features, sanitize_features
from src.utils.logger import get_logger

logger = get_logger("features.selection")


def rank_and_select_features(
    df: pd.DataFrame,
    version_name: str,
    encoding_fix_map: dict | None = None,
    top_n: int | None = None,
    min_combined_score: float | None = None,
    min_age: int = 18,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Rankea y selecciona features usando Mutual Information + Random Forest.

    Retorna:
        ranking: DataFrame con Variable, MI_Score, RF_Importance, Combined_Score
        selected_features: lista de columnas seleccionadas
        metadata: estadísticas del proceso
    """
    df = df.copy()

    if "target" not in df.columns:
        raise ValueError(f"[{version_name}] Columna 'target' no encontrada")

    # Filtro de edad
    if "Edad" in df.columns:
        df = df[df["Edad"] >= min_age].reset_index(drop=True)

    df = apply_encoding_fixes(df, encoding_fix_map)
    candidate_features = infer_candidate_features(df)
    valid_features, dropped = sanitize_features(df, candidate_features)

    if not valid_features:
        raise ValueError(f"[{version_name}] No hay features candidatas válidas")

    X = df[valid_features].apply(pd.to_numeric, errors="coerce").fillna(-1)
    y = df["target"].astype(int)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    mi_df = pd.DataFrame({"Variable": list(X.columns), "MI_Score": mi_scores})

    # Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=20,
        min_samples_leaf=10, random_state=random_state,
        n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X, y)
    rf_df = pd.DataFrame({"Variable": list(X.columns), "RF_Importance": rf.feature_importances_})

    ranking = mi_df.merge(rf_df, on="Variable", how="outer").fillna(0)
    scaler = MinMaxScaler()
    ranking["MI_Norm"] = scaler.fit_transform(ranking[["MI_Score"]])
    ranking["RF_Norm"] = scaler.fit_transform(ranking[["RF_Importance"]])
    ranking["Combined_Score"] = ranking[["MI_Norm", "RF_Norm"]].mean(axis=1)
    ranking = ranking.sort_values("Combined_Score", ascending=False).reset_index(drop=True)

    if top_n is not None:
        selected = ranking.head(top_n)
    elif min_combined_score is not None:
        selected = ranking[ranking["Combined_Score"] >= min_combined_score]
    else:
        selected = ranking
    selected_features = list(selected["Variable"])

    metadata = {
        "version": version_name,
        "rows": len(df),
        "target_pct": round(float(y.mean()) * 100, 2),
        "candidate_features": len(valid_features),
        "selected_features": len(selected_features),
        "dropped_low_score": len(ranking) - len(selected_features),
        "dropped_constant": len(dropped),
        "min_combined_score_used": min_combined_score,
    }

    logger.info(
        f"[{version_name}] Selección: {len(selected_features)} features "
        f"de {len(valid_features)} candidatas "
        f"(descartadas por score bajo: {metadata['dropped_low_score']})"
    )

    return ranking, selected_features, metadata
