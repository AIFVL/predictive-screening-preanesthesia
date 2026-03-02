"""
Pipeline de validación clínica y estadística para el proyecto de screening preanestésico.

Bloques cubiertos:
  1.2  Validar coherencia clínica de prevalencia
  2.1  Verificar separabilidad básica (AUC univariado, tests)
  2.2  Verificar estabilidad de importancia (feature stability)
  4.2  Análisis de calibración (Brier, ECE, curva)
  5.1  Cross-validation real (Stratified K-Fold)
  5.2  Análisis por subgrupos (edad, severidad, anestesia)
  6.1  Score compuesto y ranking final
  6.2  Evaluación clínica cualitativa (TP/FN/FP para revisión)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

from utils.feature_engineering import ENCODING_FIX_MAP, sanitize_features_for_subset
import utils.modeling as mod
from utils.multi_version.modeling_pipeline import (
    extract_version_from_merged,
    extract_version_from_features,
    load_features_list,
    prepare_xy_for_version,
    stratified_train_val_test_split,
    balance_binary_dataset,
    build_case_review_table,
)


# ============================================================================
# 1.2  Prevalence analysis
# ============================================================================

def run_target_prevalence_analysis(
    merged_dir: Path,
    output_dir: Path,
    active_versions: list[str] | None = None,
    severity_col: str | None = None,
    anesthesia_col: str | None = None,
) -> pd.DataFrame:
    """
    Calcula prevalencia global, por categoría de severidad y por tipo de anestesia
    para cada versión de target.

    Output: target_prevalence_analysis.csv
    Columnas: Version, N_Total, N_Positivos, Prevalencia_Global,
              Category, Category_Value, N_Cat, N_Pos_Cat, Prevalencia_Cat
    """
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    if not merged_files:
        raise FileNotFoundError("No se encontraron datasets mergeados")

    merged_map = {extract_version_from_merged(p): p for p in merged_files}
    if active_versions:
        merged_map = {k: v for k, v in merged_map.items() if k in set(active_versions)}

    # Auto-detect severity / anesthesia columns
    _severity_candidates = [
        severity_col,
        "severity_ordinal_proc",
        "predicted_label_proc_encoded",
        "ASA",
    ]
    _anesthesia_candidates = [
        anesthesia_col,
        "Tipo de anestesia",
        "Tipo_Anestesia",
    ]

    rows: list[dict] = []

    for version, path in sorted(merged_map.items()):
        df = pd.read_excel(path)
        if "Edad" in df.columns:
            df = df[df["Edad"] >= 18].reset_index(drop=True)
        df = df.rename(columns=ENCODING_FIX_MAP)

        if "target" not in df.columns:
            continue

        y = df["target"].astype(int)
        n_total = len(y)
        n_pos = int(y.sum())
        prev_global = float(y.mean())

        # Global row
        rows.append({
            "Version": version,
            "N_Total": n_total,
            "N_Positivos": n_pos,
            "Prevalencia_Global": round(prev_global, 4),
            "Category": "global",
            "Category_Value": "all",
            "N_Cat": n_total,
            "N_Pos_Cat": n_pos,
            "Prevalencia_Cat": round(prev_global, 4),
        })

        # By severity
        sev_col = _find_col(df, _severity_candidates)
        if sev_col:
            for val in sorted(df[sev_col].dropna().unique()):
                mask = df[sev_col] == val
                n_c = int(mask.sum())
                n_p = int(y[mask].sum())
                rows.append({
                    "Version": version,
                    "N_Total": n_total,
                    "N_Positivos": n_pos,
                    "Prevalencia_Global": round(prev_global, 4),
                    "Category": "severidad",
                    "Category_Value": str(val),
                    "N_Cat": n_c,
                    "N_Pos_Cat": n_p,
                    "Prevalencia_Cat": round(n_p / n_c, 4) if n_c else 0,
                })

        # By anesthesia type
        anes_col = _find_col(df, _anesthesia_candidates)
        if anes_col:
            for val in sorted(df[anes_col].dropna().unique()):
                mask = df[anes_col] == val
                n_c = int(mask.sum())
                n_p = int(y[mask].sum())
                rows.append({
                    "Version": version,
                    "N_Total": n_total,
                    "N_Positivos": n_pos,
                    "Prevalencia_Global": round(prev_global, 4),
                    "Category": "tipo_anestesia",
                    "Category_Value": str(val),
                    "N_Cat": n_c,
                    "N_Pos_Cat": n_p,
                    "Prevalencia_Cat": round(n_p / n_c, 4) if n_c else 0,
                })

        # By age group (>65 vs <=65)
        if "Edad" in df.columns:
            for label, mask in [("edad_leq_65", df["Edad"] <= 65), ("edad_gt_65", df["Edad"] > 65)]:
                n_c = int(mask.sum())
                n_p = int(y[mask].sum())
                rows.append({
                    "Version": version,
                    "N_Total": n_total,
                    "N_Positivos": n_pos,
                    "Prevalencia_Global": round(prev_global, 4),
                    "Category": "edad",
                    "Category_Value": label,
                    "N_Cat": n_c,
                    "N_Pos_Cat": n_p,
                    "Prevalencia_Cat": round(n_p / n_c, 4) if n_c else 0,
                })

    df_out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "target_prevalence_analysis.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[prevalence] Guardado: {out_path}")
    return df_out


# ============================================================================
# 2.1  Signal analysis (univariate AUC + tests)
# ============================================================================

def run_target_signal_analysis(
    merged_dir: Path,
    features_versioned_dir: Path,
    output_dir: Path,
    active_versions: list[str] | None = None,
    top_n: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Para cada versión de target:
      - Calcula AUC univariado por feature
      - Mann-Whitney U test entre clases
    Output: target_signal_analysis_<version>.csv
    """
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    feature_files = sorted(features_versioned_dir.glob("*_selected_features_list.txt"))

    merged_map = {extract_version_from_merged(p): p for p in merged_files}
    features_map = {extract_version_from_features(p): p for p in feature_files}
    versions = sorted(set(merged_map) & set(features_map))
    if active_versions:
        versions = [v for v in versions if v in set(active_versions)]

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for version in versions:
        df = pd.read_excel(merged_map[version])
        feats = load_features_list(features_map[version])
        X, y, feats = prepare_xy_for_version(df, feats)

        rows = []
        for col in feats[:top_n]:
            vals = X[col].dropna()
            idx = vals.index
            y_aligned = y.loc[idx]

            # AUC univariado
            try:
                auc_uni = roc_auc_score(y_aligned, vals)
            except ValueError:
                auc_uni = np.nan

            # Mann-Whitney U
            g0 = vals[y_aligned == 0]
            g1 = vals[y_aligned == 1]
            if len(g0) > 1 and len(g1) > 1:
                try:
                    stat, pval = stats.mannwhitneyu(g0, g1, alternative="two-sided")
                except Exception:
                    stat, pval = np.nan, np.nan
            else:
                stat, pval = np.nan, np.nan

            rows.append({
                "Feature": col,
                "AUC_Univariado": round(auc_uni, 4) if not np.isnan(auc_uni) else np.nan,
                "MannWhitney_Stat": stat,
                "p_value": pval,
                "Mean_Class0": float(g0.mean()) if len(g0) else np.nan,
                "Mean_Class1": float(g1.mean()) if len(g1) else np.nan,
            })

        df_signal = pd.DataFrame(rows).sort_values("AUC_Univariado", ascending=False, na_position="last")
        path = output_dir / f"target_signal_analysis_{version}.csv"
        df_signal.to_csv(path, index=False)
        results[version] = df_signal
        print(f"[signal] {version}: {len(df_signal)} features → {path}")

    return results


# ============================================================================
# 2.2  Feature stability across splits
# ============================================================================

def run_feature_stability_analysis(
    merged_dir: Path,
    features_versioned_dir: Path,
    output_dir: Path,
    active_versions: list[str] | None = None,
    n_splits: int = 5,
    top_k: int = 10,
    random_states: tuple[int, ...] = (42, 123, 456, 789, 2024),
) -> pd.DataFrame:
    """
    Evalúa la estabilidad de las top-K features más importantes a través de
    múltiples splits aleatorios.

    Stability_Score = promedio de % overlap del top-K entre todos los pares de splits.

    Output: feature_stability_report.csv
    """
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    feature_files = sorted(features_versioned_dir.glob("*_selected_features_list.txt"))

    merged_map = {extract_version_from_merged(p): p for p in merged_files}
    features_map = {extract_version_from_features(p): p for p in feature_files}
    versions = sorted(set(merged_map) & set(features_map))
    if active_versions:
        versions = [v for v in versions if v in set(active_versions)]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for version in versions:
        df = pd.read_excel(merged_map[version])
        feats = load_features_list(features_map[version])
        X, y, feats = prepare_xy_for_version(df, feats)

        top_sets: list[set[str]] = []

        for rs in random_states[:n_splits]:
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
                X, y, random_state=rs,
            )
            # Train a quick RF only on X_train
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=rs,
            )
            rf.fit(X_train, y_train)
            importances = pd.Series(rf.feature_importances_, index=X_train.columns)
            top_features = set(importances.nlargest(top_k).index.tolist())
            top_sets.append(top_features)

        # Pairwise overlap
        overlaps = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                overlap = len(top_sets[i] & top_sets[j]) / top_k
                overlaps.append(overlap)

        stability_score = float(np.mean(overlaps)) if overlaps else 0.0

        # Frequency of each feature across splits
        from collections import Counter
        freq = Counter()
        for s in top_sets:
            freq.update(s)
        most_stable = [f for f, c in freq.most_common(top_k)]

        report_rows.append({
            "Version": version,
            "N_Splits": n_splits,
            "Top_K": top_k,
            "Stability_Score": round(stability_score, 4),
            "Most_Stable_Features": ", ".join(most_stable[:top_k]),
        })

    df_report = pd.DataFrame(report_rows)
    path = output_dir / "feature_stability_report.csv"
    df_report.to_csv(path, index=False)
    print(f"[stability] Guardado: {path}")
    return df_report


# ============================================================================
# 4.2  Calibration analysis
# ============================================================================

def run_calibration_analysis(
    predictions_by_version: dict,
    y_test_by_version: dict,
    output_dir: Path,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Genera reporte de calibración por versión y modelo.

    Output: calibration_report_<version>_<model>.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for version, models in predictions_by_version.items():
        y_test = y_test_by_version[version]
        for model_name, pred_data in models.items():
            y_proba = pred_data["y_proba"]
            cal_metrics = mod.compute_calibration_metrics(y_test, y_proba, n_bins=n_bins)

            safe_model = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            rows = [{
                "Version": version,
                "Model": model_name,
                "Brier_Score": cal_metrics["Brier_Score"],
                "ECE": cal_metrics["ECE"],
            }]
            df_cal = pd.DataFrame(rows)
            path = output_dir / f"calibration_report_{version}_{safe_model}.csv"
            df_cal.to_csv(path, index=False)
            results[f"{version}_{model_name}"] = df_cal

    print(f"[calibration] {len(results)} reportes guardados en {output_dir}")
    return results


# ============================================================================
# 5.1  Stratified K-Fold cross-validation
# ============================================================================

def run_cross_validation(
    merged_dir: Path,
    features_versioned_dir: Path,
    output_dir: Path,
    active_versions: list[str] | None = None,
    n_folds: int = 5,
    random_state: int = 42,
    models_to_cv: list[str] | None = None,
) -> pd.DataFrame:
    """
    Ejecuta Stratified K-Fold CV para cada versión y modelos seleccionados.

    Output: cv_results_by_version.csv
    Columnas: Version, Model, Fold, ROC_AUC, F2, Recall, Precision, Balanced_Accuracy,
              Mean_ROC_AUC, Std_ROC_AUC, Mean_F2, Std_F2
    """
    merged_files = sorted(merged_dir.glob("OPERA_COMPLETO_*.xlsx"))
    feature_files = sorted(features_versioned_dir.glob("*_selected_features_list.txt"))

    merged_map = {extract_version_from_merged(p): p for p in merged_files}
    features_map = {extract_version_from_features(p): p for p in feature_files}
    versions = sorted(set(merged_map) & set(features_map))
    if active_versions:
        versions = [v for v in versions if v in set(active_versions)]

    if models_to_cv is None:
        models_to_cv = ["Random Forest", "XGBoost", "HistGradientBoosting", "Regresión Logística"]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for version in versions:
        df = pd.read_excel(merged_map[version])
        feats = load_features_list(features_map[version])
        X, y, feats = prepare_xy_for_version(df, feats)

        pos_weight_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        models_dict = mod.get_models_definitions(
            random_state=random_state,
            scale_pos_weight=pos_weight_ratio,
        )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for model_name in models_to_cv:
            if model_name not in models_dict:
                continue

            fold_metrics: list[dict] = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                model = mod.get_models_definitions(
                    random_state=random_state,
                    scale_pos_weight=pos_weight_ratio,
                )[model_name]
                model.fit(X_tr, y_tr)

                y_proba = model.predict_proba(X_te)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                try:
                    auc = roc_auc_score(y_te, y_proba)
                except ValueError:
                    auc = np.nan

                f2 = fbeta_score(y_te, y_pred, beta=2, zero_division=0)
                rec = recall_score(y_te, y_pred, zero_division=0)
                prec = precision_score(y_te, y_pred, zero_division=0)
                bal_acc = balanced_accuracy_score(y_te, y_pred)

                fold_metrics.append({
                    "ROC_AUC": auc,
                    "F2": f2,
                    "Recall": rec,
                    "Precision": prec,
                    "Balanced_Accuracy": bal_acc,
                })

                all_rows.append({
                    "Version": version,
                    "Model": model_name,
                    "Fold": fold_idx,
                    "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
                    "F2": round(f2, 4),
                    "Recall": round(rec, 4),
                    "Precision": round(prec, 4),
                    "Balanced_Accuracy": round(bal_acc, 4),
                })

            # Add summary row
            aucs = [m["ROC_AUC"] for m in fold_metrics if not np.isnan(m["ROC_AUC"])]
            f2s = [m["F2"] for m in fold_metrics]
            all_rows.append({
                "Version": version,
                "Model": model_name,
                "Fold": "mean",
                "ROC_AUC": round(np.mean(aucs), 4) if aucs else np.nan,
                "F2": round(np.mean(f2s), 4),
                "Recall": round(np.mean([m["Recall"] for m in fold_metrics]), 4),
                "Precision": round(np.mean([m["Precision"] for m in fold_metrics]), 4),
                "Balanced_Accuracy": round(np.mean([m["Balanced_Accuracy"] for m in fold_metrics]), 4),
            })
            all_rows.append({
                "Version": version,
                "Model": model_name,
                "Fold": "std",
                "ROC_AUC": round(np.std(aucs), 4) if aucs else np.nan,
                "F2": round(np.std(f2s), 4),
                "Recall": round(np.std([m["Recall"] for m in fold_metrics]), 4),
                "Precision": round(np.std([m["Precision"] for m in fold_metrics]), 4),
                "Balanced_Accuracy": round(np.std([m["Balanced_Accuracy"] for m in fold_metrics]), 4),
            })

        print(f"[cv] {version}: {n_folds}-fold CV completo para {len(models_to_cv)} modelos")

    df_cv = pd.DataFrame(all_rows)
    path = output_dir / "cv_results_by_version.csv"
    df_cv.to_csv(path, index=False)
    print(f"[cv] Guardado: {path}")
    return df_cv


# ============================================================================
# 5.2  Subgroup analysis
# ============================================================================

def run_subgroup_analysis(
    predictions_by_version: dict,
    y_test_by_version: dict,
    x_test_by_version: dict,
    output_dir: Path,
    best_model_by_version: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Evalúa métricas del mejor modelo por subgrupos:
      - Edad (>65 vs ≤65)
      - Severidad alta vs baja (si existe columna)
      - Tipo de anestesia (si existe columna)

    Output: subgroup_analysis_<version>.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for version, models in predictions_by_version.items():
        y_test = y_test_by_version[version]
        X_test = x_test_by_version[version]

        # Select best model for this version
        if best_model_by_version and version in best_model_by_version:
            best_name = best_model_by_version[version]
        else:
            best_name = list(models.keys())[0]

        pred_data = models[best_name]
        y_pred = pred_data["y_pred"]
        y_proba = pred_data["y_proba"]

        rows: list[dict] = []

        # Age subgroups
        if "Edad" in X_test.columns:
            _add_subgroup_metrics(rows, version, best_name, "edad",
                                  "leq_65", X_test["Edad"] <= 65,
                                  y_test, y_pred, y_proba)
            _add_subgroup_metrics(rows, version, best_name, "edad",
                                  "gt_65", X_test["Edad"] > 65,
                                  y_test, y_pred, y_proba)

        # Severity subgroups
        sev_col = _find_col(X_test, [
            "severity_ordinal_proc", "predicted_label_proc_encoded", "ASA",
        ])
        if sev_col:
            median_val = X_test[sev_col].median()
            _add_subgroup_metrics(rows, version, best_name, "severidad",
                                  "baja", X_test[sev_col] <= median_val,
                                  y_test, y_pred, y_proba)
            _add_subgroup_metrics(rows, version, best_name, "severidad",
                                  "alta", X_test[sev_col] > median_val,
                                  y_test, y_pred, y_proba)

        # Anesthesia type subgroups
        anes_col = _find_col(X_test, ["Tipo de anestesia", "Tipo_Anestesia"])
        if anes_col:
            for val in X_test[anes_col].dropna().unique():
                mask = X_test[anes_col] == val
                _add_subgroup_metrics(rows, version, best_name, "tipo_anestesia",
                                      str(val), mask,
                                      y_test, y_pred, y_proba)

        df_sub = pd.DataFrame(rows)
        path = output_dir / f"subgroup_analysis_{version}.csv"
        df_sub.to_csv(path, index=False)
        results[version] = df_sub
        print(f"[subgroup] {version}: {len(df_sub)} filas → {path}")

    return results


# ============================================================================
# 6.1  Final composite score & ranking
# ============================================================================

def compute_final_ranking(
    baseline_results_path: Path,
    cv_results_path: Path,
    stability_report_path: Path,
    output_dir: Path,
    calibration_dir: Path | None = None,
    weight_roc_auc: float = 0.35,
    weight_f2: float = 0.35,
    weight_stability: float = 0.20,
    weight_calibration: float = 0.10,
) -> pd.DataFrame:
    """
    Calcula un score compuesto por versión y genera ranking final.

    Score_Final = w_auc * ROC-AUC_norm + w_f2 * F2_norm
                + w_stab * Stability_score + w_cal * Calibration_score

    Output: final_target_ranking.csv
    """
    df_baseline = pd.read_csv(baseline_results_path)
    df_cv = pd.read_csv(cv_results_path)
    df_stability = pd.read_csv(stability_report_path)

    # Best model per version from baseline
    best_per_version = (
        df_baseline.sort_values(["Version", "ROC-AUC", "F2"], ascending=[True, False, False])
        .groupby("Version", as_index=False)
        .first()
    )

    # CV mean ROC-AUC per version (from summary rows)
    cv_means = df_cv[df_cv["Fold"] == "mean"].copy()
    cv_best = (
        cv_means.sort_values(["Version", "ROC_AUC"], ascending=[True, False])
        .groupby("Version", as_index=False)
        .first()
    )

    # Merge
    merged = best_per_version[["Version", "ROC-AUC", "F2", "Recall", "Precision", "FN_Rate"]].copy()
    merged = merged.rename(columns={"ROC-AUC": "ROC_AUC_Test"})

    if not cv_best.empty:
        merged = merged.merge(
            cv_best[["Version", "ROC_AUC", "F2"]].rename(
                columns={"ROC_AUC": "ROC_AUC_CV", "F2": "F2_CV"}
            ),
            on="Version",
            how="left",
        )
    else:
        merged["ROC_AUC_CV"] = merged["ROC_AUC_Test"]
        merged["F2_CV"] = merged["F2"]

    merged = merged.merge(
        df_stability[["Version", "Stability_Score"]],
        on="Version",
        how="left",
    )
    merged["Stability_Score"] = merged["Stability_Score"].fillna(0)

    # Calibration score (1 - ECE, so higher is better)
    if calibration_dir and calibration_dir.exists():
        cal_rows = []
        for cal_file in calibration_dir.glob("calibration_report_*.csv"):
            try:
                df_cal = pd.read_csv(cal_file)
                cal_rows.append(df_cal)
            except Exception:
                pass
        if cal_rows:
            df_all_cal = pd.concat(cal_rows, ignore_index=True)
            cal_by_version = df_all_cal.groupby("Version")["ECE"].mean().reset_index()
            cal_by_version["Calibration_Score"] = 1 - cal_by_version["ECE"]
            merged = merged.merge(
                cal_by_version[["Version", "Calibration_Score"]],
                on="Version",
                how="left",
            )
        else:
            merged["Calibration_Score"] = 0.5
    else:
        merged["Calibration_Score"] = 0.5

    merged["Calibration_Score"] = merged["Calibration_Score"].fillna(0.5)

    # Normalize 0-1
    def _norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / rng

    merged["ROC_AUC_norm"] = _norm(merged["ROC_AUC_CV"])
    merged["F2_norm"] = _norm(merged["F2_CV"])

    merged["Score_Final"] = (
        weight_roc_auc * merged["ROC_AUC_norm"]
        + weight_f2 * merged["F2_norm"]
        + weight_stability * merged["Stability_Score"]
        + weight_calibration * merged["Calibration_Score"]
    )

    merged = merged.sort_values("Score_Final", ascending=False).reset_index(drop=True)
    merged["Rank"] = range(1, len(merged) + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "final_target_ranking.csv"
    merged.to_csv(path, index=False)
    print(f"[ranking] Guardado: {path}")
    return merged


# ============================================================================
# 6.2  Clinical review table (10 TP, 10 FN, 10 FP)
# ============================================================================

def generate_clinical_review_cases(
    predictions_by_version: dict,
    y_test_by_version: dict,
    x_test_by_version: dict,
    explainability_global_by_version: dict,
    output_dir: Path,
    cases_per_group: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Genera automáticamente 10 TP, 10 FN, 10 FP para revisión clínica manual.
    No seleccionar versión final sin revisión clínica de FN.

    Output: clinical_review_<version>.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for version, models in predictions_by_version.items():
        y_test = y_test_by_version[version]
        X_test = x_test_by_version[version]

        # Use best model (first in dict or highest AUC)
        best_name = list(models.keys())[0]
        pred_data = models[best_name]
        y_pred = pred_data["y_pred"]
        y_proba = pred_data["y_proba"]

        top_features = []
        if version in explainability_global_by_version:
            top_features = explainability_global_by_version[version]["Feature"].tolist()

        cases = build_case_review_table(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            top_features=top_features,
            max_cases_per_group=cases_per_group,
        )

        path = output_dir / f"clinical_review_{version}.csv"
        cases.to_csv(path, index=False)
        results[version] = cases
        print(f"[clinical_review] {version}: {len(cases)} casos → {path}")

    return results


# ============================================================================
# Helpers
# ============================================================================

def _find_col(df: pd.DataFrame, candidates: list[str | None]) -> str | None:
    """Retorna la primera columna candidate que existe en df, o None."""
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _add_subgroup_metrics(
    rows: list[dict],
    version: str,
    model_name: str,
    category: str,
    value: str,
    mask: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """Agrega una fila de métricas de subgrupo a rows."""
    idx = mask[mask].index
    if len(idx) < 5:
        return

    y_t = y_test.loc[idx]
    y_p = y_pred[mask.values] if hasattr(y_pred, '__getitem__') else y_pred
    y_pb = y_proba[mask.values] if hasattr(y_proba, '__getitem__') else y_proba

    # Reconstruct aligned arrays
    mask_arr = mask.values.astype(bool)
    y_p = np.asarray(y_pred)[mask_arr]
    y_pb = np.asarray(y_proba)[mask_arr]
    y_t = y_test.values[mask_arr]

    try:
        auc = roc_auc_score(y_t, y_pb)
    except ValueError:
        auc = np.nan

    rows.append({
        "Version": version,
        "Model": model_name,
        "Category": category,
        "Value": value,
        "N": int(mask_arr.sum()),
        "N_Pos": int(y_t.sum()),
        "Prevalence": round(float(y_t.mean()), 4),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "F2": round(fbeta_score(y_t, y_p, beta=2, zero_division=0), 4),
        "Recall": round(recall_score(y_t, y_p, zero_division=0), 4),
        "Precision": round(precision_score(y_t, y_p, zero_division=0), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_t, y_p), 4),
    })
