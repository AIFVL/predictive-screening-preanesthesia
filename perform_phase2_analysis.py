"""
Script de ejecución directa para Fase 2 de Modelado.
FILTRO: Solo pacientes adultos (Edad >= 18).
"""
import sys
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sys.path.append(os.path.abspath('.'))

from utils.data_loader import load_opera_completo, load_features_metadata
from utils.feature_engineering import resolve_feature_columns, get_imputation_strategies, ENCODING_FIX_MAP
import utils.modeling as mod
import utils.visualization as viz

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import shap

def run_analysis():
    print("Iniciando Fase 2: Deep Model Optimization (Solo Adultos)...")
    
    # 1. Carga
    print("\n[1] Cargando datos...")
    df = load_opera_completo()
    print(f"    Dataset completo: {len(df):,} registros")
    
    # FILTRO ADULTOS
    n_antes = len(df)
    df = df[df['Edad'] >= 18].reset_index(drop=True)
    print(f"    Filtro adultos (Edad >= 18): {n_antes:,} -> {len(df):,}")
    print(f"    Pacientes pediatricos excluidos: {n_antes - len(df):,}")
    
    features_meta = load_features_metadata()
    selected_names = features_meta['Variable'].tolist()
    feature_columns, missing = resolve_feature_columns(selected_names, df.columns, features_meta)
    
    df = df.rename(columns=ENCODING_FIX_MAP)
    feature_columns = [ENCODING_FIX_MAP.get(c, c) for c in feature_columns]
    
    X = df[feature_columns].copy()
    y = df['target'].copy()
    print(f"    X: {X.shape}, y: {y.shape}")
    
    # 2. Split
    print("\n[2] Split y Smart Imputation...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    pos_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    strategies = get_imputation_strategies(feature_columns)
    cols_zero = strategies['fill_zero']
    cols_median = strategies['fill_median']
    
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ('zero', SimpleImputer(strategy='constant', fill_value=0), cols_zero),
            ('median', SimpleImputer(strategy='median'), cols_median)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ('zero', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ]), cols_zero),
            ('median', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), cols_median)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X_train_tree = preprocessor_tree.fit_transform(X_train)
    X_test_tree = preprocessor_tree.transform(X_test)
    X_train_linear = preprocessor_linear.fit_transform(X_train)
    X_test_linear = preprocessor_linear.transform(X_test)
    
    # 3. Modelado
    print("\n[3] Entrenando modelos base...")
    models_dict = mod.get_models_definitions(random_state=RANDOM_STATE, scale_pos_weight=pos_weight_ratio)
    results = {}
    predictions = {}
    
    for name, model in models_dict.items():
        print(f"    {name}...")
        if name == 'Regresión Logística':
            X_tr, X_te = X_train_linear, X_test_linear
        else:
            X_tr, X_te = X_train_tree, X_test_tree
        y_pred, y_proba, metrics = mod.evaluate_model_performance(name, model, X_tr, X_te, y_train, y_test)
        results[name] = metrics
        predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}
        
    df_results = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
    print("\n    RANKING POR ROC-AUC (Solo Adultos):")
    print(df_results)
    
    viz.plot_confusion_matrices(predictions, y_test, save_path='matrices_confusion_phase2.png')
    viz.plot_roc_pr_curves(predictions, y_test, save_path='curvas_roc_pr_phase2.png')
    
    # 4. Optimización
    best_model_name = df_results.index[0]
    best_auc = df_results.iloc[0]['ROC-AUC']
    print(f"\n[4] Optimizando: {best_model_name} (AUC: {best_auc:.4f})")
    
    if best_model_name == 'Regresión Logística':
        X_opt, X_test_opt = X_train_linear, X_test_linear
    else:
        X_opt, X_test_opt = X_train_tree, X_test_tree
        
    study = mod.optimize_best_model(best_model_name, X_opt, y_train, random_state=RANDOM_STATE, scale_pos_weight=pos_weight_ratio, n_trials=30, metric='roc_auc')
    best_params = study.best_params
    print(f"    Mejor Score CV: {study.best_value:.4f}")
    
    final_model = None
    if best_model_name == 'XGBoost':
        final_model = xgb.XGBClassifier(**best_params, scale_pos_weight=pos_weight_ratio, eval_metric='aucpr', use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1)
    elif best_model_name == 'HistGradientBoosting':
        final_model = HistGradientBoostingClassifier(**best_params, class_weight='balanced', scoring='roc_auc', random_state=RANDOM_STATE)
    elif best_model_name == 'Random Forest':
        final_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    elif best_model_name == 'Regresión Logística':
        final_model = LogisticRegression(**best_params, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    
    y_pred_opt, y_proba_opt, metrics_opt = mod.evaluate_model_performance(f'{best_model_name} Optimizado', final_model, X_opt, X_test_opt, y_train, y_test)
    
    opt_thresh, opt_f2, thresholds, f2_scores = mod.find_optimal_threshold(y_test, y_proba_opt)
    viz.plot_threshold_optimization(thresholds, f2_scores, opt_thresh, save_path='umbral_optimo_phase2.png')
    print(f"    Umbral F2: {opt_thresh:.2f} -> F2: {opt_f2:.4f}")
    
    # 5. SHAP
    print("\n[5] SHAP...")
    try:
        if best_model_name in ['XGBoost', 'Random Forest', 'HistGradientBoosting']:
            explainer = shap.TreeExplainer(final_model)
            shap_explanation = explainer(X_test_opt)
        else:
            explainer = shap.Explainer(final_model, X_opt)
            shap_explanation = explainer(X_test_opt)
        vals = shap_explanation.values
        if len(vals.shape) == 3: vals = vals[:, :, 1]
        viz.plot_shap_summary(vals, X_test_opt, save_path='shap_summary_phase2.png')
        shap_importance = np.abs(vals).mean(axis=0)
        top_indices = np.argsort(shap_importance)[-5:][::-1]
        top_features = [X_test_opt.columns[i] for i in top_indices]
        viz.plot_shap_dependence(vals, X_test_opt, top_features, save_path='shap_dependence_phase2.png')
    except Exception as e:
        print(f"    Error en SHAP: {e}")
        
    print("\nAnalisis completado (Solo Adultos).")

if __name__ == "__main__":
    run_analysis()
