# src/models/hyperparameter_search.py
from __future__ import annotations

import pandas as pd
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.models.registry import build_model
from src.utils.logger import get_logger

logger = get_logger("models.hyperparameter_search")


def search_hyperparameters(
    model_cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 70,
    scoring: str = "f2",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Búsqueda de hiperparámetros con Optuna sobre el search_space de model_cfg.

    Retorna best_params dict con los parámetros óptimos encontrados.
    Solo se llama cuando hyperparameter_search.enabled=true en pipeline_config.yaml.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_space = model_cfg.get("search_space", {})
    if not search_space:
        logger.info("No hay search_space definido — usando params base")
        return dict(model_cfg.get("params", {}))

    scorer = make_scorer(fbeta_score, beta=2, zero_division=0) if scoring == "f2" else scoring
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(-1)

    def objective(trial):
        trial_params = {}
        for param_name, values in search_space.items():
            if isinstance(values, list):
                clean_values = [v for v in values if v is not None]
                trial_params[param_name] = trial.suggest_categorical(param_name, clean_values)
            elif isinstance(values, dict) and "low" in values:
                trial_params[param_name] = trial.suggest_float(
                    param_name, values["low"], values["high"],
                    log=values.get("log", False)
                )
        cfg_trial = {**model_cfg, "params": {**model_cfg.get("params", {}), **trial_params}}
        model = build_model(cfg_trial, random_state=random_state)
        scores = cross_val_score(model, X_num, y, cv=cv, scoring=scorer, n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

    best_params = {**model_cfg.get("params", {}), **study.best_params}
    logger.info(f"Mejor score ({scoring}): {study.best_value:.4f} | params: {study.best_params}")
    return best_params
