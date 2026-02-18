import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, fbeta_score,
    roc_auc_score, brier_score_loss, confusion_matrix, 
    roc_curve, precision_recall_curve, make_scorer
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import optuna


def _get_scorer(metric='roc_auc'):
    """Retorna el scorer de sklearn según el metric solicitado."""
    if metric == 'roc_auc':
        return 'roc_auc'
    elif metric == 'f2':
        return make_scorer(fbeta_score, beta=2)
    else:
        return metric


def get_models_definitions(random_state=42, scale_pos_weight=1.0):
    """
    Retorna un diccionario con los modelos base a entrenar.
    Modelos Clínicos Profesionales:
    - LR: Baseline interpretable.
    - RF: Robusto, no lineal.
    - XGB: Boosting clásico.
    - HistGB: SOTA para datos tabulares con missings.
    """
    models = {
        'Regresión Logística': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ),
        'Árbol de Decisión': DecisionTreeClassifier(
            max_depth=5,
            class_weight='balanced',
            min_samples_leaf=50,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state
        ),
        'XGBoost': xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            learning_rate=0.1,
            n_estimators=500,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            class_weight='balanced',
            scoring='roc_auc',
            random_state=random_state,
            max_iter=500
        )
    }
    return models


def evaluate_model_performance(name, model, X_tr, X_te, y_tr, y_te):
    """
    Entrena y evalúa un modelo.
    """
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    
    metrics = {
        'Recall': recall_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred),
        'F1': f1_score(y_te, y_pred),
        'F2': fbeta_score(y_te, y_pred, beta=2),
        'ROC-AUC': roc_auc_score(y_te, y_proba),
        'Brier': brier_score_loss(y_te, y_proba),
    }
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
        
    return y_pred, y_proba, metrics


def optimize_best_model(model_name, X_train, y_train, random_state=42, scale_pos_weight=1.0, n_trials=50, metric='roc_auc'):
    """
    Despacha la optimización maximizando ROC-AUC (mejor ranking) o F2 (mejor decisión).
    Recomendación: Optimizar ROC-AUC para robustez, luego ajustar umbral.
    """
    print(f"Optimizing hyperparameters for: {model_name} (Target: {metric})")
    
    if model_name == 'XGBoost':
        return train_optimized_xgboost(X_train, y_train, random_state, scale_pos_weight, n_trials, metric)
    elif model_name == 'Random Forest':
        return optimize_random_forest(X_train, y_train, random_state, n_trials, metric)
    elif model_name == 'HistGradientBoosting':
        return optimize_hist_gradient_boosting(X_train, y_train, random_state, n_trials, metric)
    elif model_name == 'Árbol de Decisión':
        return optimize_decision_tree(X_train, y_train, random_state, n_trials, metric)
    elif model_name == 'Regresión Logística':
        return optimize_logistic_regression(X_train, y_train, random_state, n_trials, metric)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado para optimización.")


def train_optimized_xgboost(X_train, y_train, random_state=42, scale_pos_weight=1.0, n_trials=50, metric='roc_auc'):
    scorer = _get_scorer(metric)
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1
        )
        return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_random_forest(X_train, y_train, random_state=42, n_trials=50, metric='roc_auc'):
    scorer = _get_scorer(metric)
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        model = RandomForestClassifier(
            **params,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_hist_gradient_boosting(X_train, y_train, random_state=42, n_trials=50, metric='roc_auc'):
    scorer = _get_scorer(metric)
    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
        }
        model = HistGradientBoostingClassifier(
            **params,
            class_weight='balanced',
            scoring='roc_auc',
            random_state=random_state
        )
        return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_decision_tree(X_train, y_train, random_state=42, n_trials=50, metric='roc_auc'):
    scorer = _get_scorer(metric)
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        }
        model = DecisionTreeClassifier(
            **params,
            class_weight='balanced',
            random_state=random_state
        )
        return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_logistic_regression(X_train, y_train, random_state=42, n_trials=50, metric='roc_auc'):
    scorer = _get_scorer(metric)
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 100.0, log=True)
        model = LogisticRegression(
            C=C,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )
        return cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study


def find_optimal_threshold(y_true, y_proba):
    """
    Encuentra el umbral de decisión que maximiza F2-Score.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f2_scores = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2)
        f2_scores.append(f2)
        
    best_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]
    
    return optimal_threshold, best_f2, thresholds, f2_scores
