# src/models/registry.py
from __future__ import annotations

import importlib
import inspect

from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import get_logger

logger = get_logger("models.registry")


def _instantiate(module_name: str, class_name: str, params: dict, random_state: int):
    """Instancia una clase desde module+class+params, inyectando random_state si aplica."""
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    p = dict(params)
    sig = inspect.signature(cls.__init__)
    if "random_state" in sig.parameters and "random_state" not in p:
        p["random_state"] = random_state
    logger.info(f"Instanciando {class_name} con params: {p}")
    return cls(**p)


def build_model(model_cfg: dict, random_state: int = 42):
    """
    Instancia un modelo a partir de su config dict.

    Para modelos simples:
        module, class, params

    Para StackingClassifier / VotingClassifier:
        module, class, params, estimators (lista de nombres),
        _estimator_configs (dict nombre->cfg), meta_estimator (nombre, solo Stacking)

    Si calibrate=True, envuelve el modelo en CalibratedClassifierCV(method='isotonic').
    Los ensambles nunca se calibran (calibrate ignorado para ellos).
    """
    module_name = model_cfg["module"]
    class_name = model_cfg["class"]
    params = dict(model_cfg.get("params", {}))
    calibrate = model_cfg.get("calibrate", False)

    if class_name in ("StackingClassifier", "VotingClassifier"):
        estimator_names = model_cfg.get("estimators", [])
        estimator_cfgs = model_cfg.get("_estimator_configs", {})

        base_estimators = []
        for name in estimator_names:
            cfg = estimator_cfgs[name]
            est = _instantiate(cfg["module"], cfg["class"],
                               cfg.get("params", {}), random_state)
            base_estimators.append((name, est))

        if class_name == "StackingClassifier":
            meta_name = model_cfg.get("meta_estimator")
            meta_cfg = estimator_cfgs[meta_name]
            meta = _instantiate(meta_cfg["module"], meta_cfg["class"],
                                meta_cfg.get("params", {}), random_state)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(estimators=base_estimators, final_estimator=meta, **params)
        else:  # VotingClassifier
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(estimators=base_estimators, **params)

    model = _instantiate(module_name, class_name, params, random_state)

    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=5)

    return model
