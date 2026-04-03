import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, fbeta_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    balanced_accuracy_score, average_precision_score,
    roc_curve, precision_recall_curve, make_scorer
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import optuna
from copy import deepcopy


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
    base_lr = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )

    base_dt = DecisionTreeClassifier(
        max_depth=5,
        class_weight='balanced',
        min_samples_leaf=50,
        random_state=random_state
    )

    base_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state
    )

    base_xgb = xgb.XGBClassifier(
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
    )

    base_hist = HistGradientBoostingClassifier(
        class_weight='balanced',
        random_state=random_state,
        max_iter=500
    )

    base_extra = ExtraTreesClassifier(
        n_estimators=500,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )

    base_gb = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        max_depth=3,
        random_state=random_state,
    )

    base_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )

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
            random_state=random_state,
            max_iter=500
        ),
        'Extra Trees': base_extra,
        'Gradient Boosting': base_gb,
        'MLP (Red Neuronal)': base_mlp,
        'Voting Ensemble': VotingClassifier(
            estimators=[
                ('lr', base_lr),
                ('rf', base_rf),
                ('xgb', base_xgb),
            ],
            voting='soft',
            n_jobs=-1,
        ),
        'Stacking Ensemble': StackingClassifier(
            estimators=[
                ('lr', base_lr),
                ('rf', base_rf),
                ('xgb', base_xgb),
                ('hist', base_hist),
            ],
            final_estimator=LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state,
            ),
            stack_method='predict_proba',
            cv=5,
            n_jobs=-1,
            passthrough=False,
        ),
    }
    return models


def evaluate_model_performance(
    name, model, X_tr, X_te, y_tr, y_te,
    optimize_threshold=True,
    recall_min: float = 0.80,
):
    """
    Entrena y evalúa un modelo.

    Parámetros:
        optimize_threshold: Si True, encuentra el threshold óptimo.
                            Si False, usa threshold=0.5 (default).
        recall_min: Recall mínimo garantizado al optimizar el threshold.
                    El threshold se elige maximizando Precision sujeto a
                    Recall >= recall_min, evitando el clasificador trivial
                    que predice siempre positivo.
    """
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]

    # Encontrar threshold óptimo si se solicita
    if optimize_threshold:
        optimal_threshold, best_score, thresholds, scores = find_optimal_threshold(
            y_te, y_proba,
            optimize_for='recall_constraint',
            recall_min=recall_min,
        )
        y_pred = (y_proba >= optimal_threshold).astype(int)
        threshold_used = optimal_threshold
    else:
        y_pred = model.predict(X_te)  # Usa threshold=0.5 por defecto
        threshold_used = 0.5

    metrics = compute_classification_metrics(
        y_true=y_te,
        y_pred=y_pred,
        y_proba=y_proba,
        threshold=threshold_used,
    )

    print(f"\n{'='*60}")
    print(f"  {name}")
    if optimize_threshold:
        print(f"  Threshold óptimo: {threshold_used:.3f} (Recall ≥ {recall_min:.0%}, maximiza Precision)")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if k != 'Threshold':
            print(f"  {k:12s}: {v:.4f}")

    return y_pred, y_proba, metrics


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba,
    threshold: float,
) -> dict:
    """
    Calcula métricas completas de clasificación binaria.
    Incluye: Accuracy, Recall, Precision, F1, F2, ROC-AUC, PR-AUC,
    Balanced Accuracy, Specificity, Brier Score, Predicted Positive Rate.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    predicted_positive_rate = float(np.mean(y_pred))
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'F2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Specificity': specificity,
        'Brier': brier_score_loss(y_true, y_proba),
        'Predicted_Positive_Rate': predicted_positive_rate,
        'FN_Rate': fn_rate,
        'Threshold': float(threshold),
    }


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


def find_optimal_threshold(
    y_true,
    y_proba,
    metric='f2',
    min_predicted_positive_rate: float | None = None,
    max_predicted_positive_rate: float | None = None,
    threshold_restrict_prevalence: bool = True,
    optimize_for: str | None = None,
    recall_min: float = 0.80,
):
    """
    Encuentra el umbral de decisión que maximiza una métrica de clasificación.

    Parámetros:
        metric: 'f2' | 'f1' | 'balanced_accuracy'. Default 'f2' (recall pesa 4x más que precision).
        min_predicted_positive_rate: límite inferior (solo si threshold_restrict_prevalence=True).
        max_predicted_positive_rate: límite superior (solo si threshold_restrict_prevalence=True).
        threshold_restrict_prevalence: si False, ignora min/max_predicted_positive_rate.
        optimize_for: si 'recall_constraint', garantiza Recall >= recall_min y maximiza
                      Precision bajo esa restricción (modo clínico recomendado).
        recall_min: recall mínimo aceptable (solo para optimize_for='recall_constraint').
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = []
    valid_thresholds = []

    # Modo recall_constraint: Recall >= recall_min, maximizar Precision
    if optimize_for == "recall_constraint":
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec >= recall_min:
                prec = precision_score(y_true, y_pred, zero_division=0)
                valid_thresholds.append(t)
                scores.append(prec)
        if not scores:
            # Fallback: threshold que maximice recall
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                rec = recall_score(y_true, y_pred, zero_division=0)
                valid_thresholds.append(t)
                scores.append(rec)
        best_idx = np.argmax(scores)
        return valid_thresholds[best_idx], scores[best_idx], np.asarray(valid_thresholds), scores

    # Modo estándar
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if threshold_restrict_prevalence:
            positive_rate = float(np.mean(y_pred))
            if min_predicted_positive_rate is not None and positive_rate < min_predicted_positive_rate:
                continue
            if max_predicted_positive_rate is not None and positive_rate > max_predicted_positive_rate:
                continue

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric debe ser 'f1', 'f2' o 'balanced_accuracy'")

        valid_thresholds.append(t)
        scores.append(score)

    if not scores:
        # Fallback seguro: sin filtros de prevalencia para no romper ejecución.
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'f2':
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                raise ValueError("metric debe ser 'f1', 'f2' o 'balanced_accuracy'")
            valid_thresholds.append(t)
            scores.append(score)

    best_idx = np.argmax(scores)
    optimal_threshold = valid_thresholds[best_idx]
    best_score = scores[best_idx]

    return optimal_threshold, best_score, np.asarray(valid_thresholds), scores


def compute_calibration_metrics(
    y_true,
    y_proba,
    n_bins: int = 10,
) -> dict:
    """
    Calcula métricas de calibración: Brier Score, ECE, curva de calibración.
    """
    brier = brier_score_loss(y_true, y_proba)

    # ECE (Expected Calibration Error)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    total = len(y_true)
    ece = 0.0
    for i in range(len(fraction_of_positives)):
        weight = bin_counts[i] / total if total > 0 else 0
        ece += weight * abs(fraction_of_positives[i] - mean_predicted_value[i])

    return {
        'Brier_Score': brier,
        'ECE': ece,
        'Calibration_Fraction_Positives': fraction_of_positives.tolist(),
        'Calibration_Mean_Predicted': mean_predicted_value.tolist(),
    }


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
    3) permutation importance (fallback genérico)
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


def instantiate_model_from_params(
    model_name: str,
    best_params: dict,
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
):
    """
    Instancia un modelo a partir de hiperparámetros optimizados.
    """
    params = deepcopy(best_params) if best_params is not None else {}

    if model_name == 'XGBoost':
        return xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == 'Random Forest':
        return RandomForestClassifier(
            **params,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(
            **params,
            class_weight='balanced',
            random_state=random_state,
        )

    if model_name == 'Árbol de Decisión':
        return DecisionTreeClassifier(
            **params,
            class_weight='balanced',
            random_state=random_state,
        )

    if model_name == 'Regresión Logística':
        return LogisticRegression(
            **params,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state,
        )

    raise ValueError(f"Modelo '{model_name}' no soportado para instanciación desde hiperparámetros")
