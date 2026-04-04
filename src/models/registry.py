# src/models/registry.py
from __future__ import annotations

import importlib
import inspect

from src.utils.logger import get_logger

logger = get_logger("models.registry")


def build_model(model_cfg: dict, random_state: int = 42):
    """
    Instancia un modelo sklearn/xgboost a partir de su config dict.

    model_cfg debe tener:
        module: "sklearn.ensemble"
        class: "RandomForestClassifier"
        params: {n_estimators: 300, ...}

    Inyecta random_state si el modelo lo acepta y no está ya en params.
    """
    module_name = model_cfg["module"]
    class_name = model_cfg["class"]
    params = dict(model_cfg.get("params", {}))

    # Reemplazar null de YAML (None) en params — mantener como None (sklearn lo acepta)
    params = {k: v for k, v in params.items()}

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    sig = inspect.signature(cls.__init__)
    if "random_state" in sig.parameters and "random_state" not in params:
        params["random_state"] = random_state

    logger.info(f"Instanciando {class_name} con params: {params}")
    return cls(**params)
