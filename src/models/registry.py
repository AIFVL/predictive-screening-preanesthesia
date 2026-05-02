# src/models/registry.py
from __future__ import annotations

import importlib
import inspect
from typing import Any

from sklearn.base import is_classifier
from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import get_logger

logger = get_logger("models.registry")


# Política de calibración por algoritmo.
# - isotonic: no paramétrico, mejor para árboles/ensembles que tienden a probabilidades sesgadas en colas.
# - sigmoid (Platt): paramétrico, suficiente cuando la salida ya es ~lineal en log-odds.
ALGORITHM_CALIBRATION_METHODS: dict[str, str] = {
    "LogisticRegression": "sigmoid",
    "MLPClassifier": "sigmoid",
    "RandomForestClassifier": "isotonic",
    "ExtraTreesClassifier": "isotonic",
    "HistGradientBoostingClassifier": "isotonic",
    "XGBClassifier": "isotonic",
    "LGBMClassifier": "isotonic",
    "StackingClassifier": "isotonic",
    "VotingClassifier": "isotonic",
}

CALIBRATION_FALLBACK_ORDER: list[str] = ["isotonic", "sigmoid"]


def _patch_class_tags(cls) -> None:
    """
    Parchea __sklearn_tags__ a nivel de CLASE para que sklearn 1.8+ reconozca
    correctamente los clasificadores de terceros (xgboost, lightgbm).

    El patch de instancia se pierde al serializar con joblib o al clonar con
    sklearn.clone(). El patch de clase persiste para todas las instancias de
    la sesión, incluyendo las cargadas de disco.
    """
    if getattr(cls, "_sklearn_compat_patched", False):
        return  # Ya parcheado en esta sesión

    original_fn = getattr(cls, "__sklearn_tags__", None)
    if original_fn is None:
        return

    def __sklearn_tags__(self):
        tags = original_fn(self)
        tags.estimator_type = "classifier"
        return tags

    cls.__sklearn_tags__ = __sklearn_tags__
    cls._sklearn_compat_patched = True


def _apply_compat_patches() -> None:
    """
    Aplica patches de compatibilidad sklearn 1.8+ al importar este módulo.
    Garantiza que cualquier instancia cargada de disco o clonada también sea
    reconocida como clasificador sin necesidad de volver a llamar a _instantiate.
    """
    for pkg, cls_name in [("xgboost", "XGBClassifier"), ("lightgbm", "LGBMClassifier")]:
        try:
            mod = importlib.import_module(pkg)
            cls = getattr(mod, cls_name, None)
            if cls is not None and not is_classifier(cls()):
                _patch_class_tags(cls)
        except Exception:
            pass 


_apply_compat_patches()


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


def _resolve_calibration_method(class_name: str, override: str | None = None) -> str:
    """Selecciona el método de calibración por algoritmo. `override` lo fuerza si se pasa."""
    if override:
        return override
    return ALGORITHM_CALIBRATION_METHODS.get(class_name, "isotonic")


def _build_base_model(model_cfg: dict, random_state: int):
    """
    Construye el estimador base (sin calibración).
    Para ensembles, instancia recursivamente sus sub-estimadores.
    """
    module_name = model_cfg["module"]
    class_name = model_cfg["class"]
    params = dict(model_cfg.get("params", {}))

    if class_name in ("StackingClassifier", "VotingClassifier"):
        estimator_names = model_cfg.get("estimators", [])
        estimator_cfgs = model_cfg.get("_estimator_configs", {})

        base_estimators = []
        for name in estimator_names:
            cfg = estimator_cfgs[name]
            est = _instantiate(cfg["module"], cfg["class"],
                               cfg.get("params", {}), random_state)
            base_estimators.append((name, est))

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        if class_name == "StackingClassifier":
            meta_name = model_cfg.get("meta_estimator")
            meta_cfg = estimator_cfgs[meta_name]
            meta = _instantiate(meta_cfg["module"], meta_cfg["class"],
                                meta_cfg.get("params", {}), random_state)
            return cls(estimators=base_estimators, final_estimator=meta, **params)
        return cls(estimators=base_estimators, **params)

    return _instantiate(module_name, class_name, params, random_state)


def _wrap_calibrated(base_model, method: str, cv: int = 5) -> CalibratedClassifierCV:
    """Envuelve un estimador en CalibratedClassifierCV con el método dado."""
    return CalibratedClassifierCV(base_model, method=method, cv=cv)


def build_model(model_cfg: dict, random_state: int = 42):
    """
    Construye un modelo (calibrado o no) sin entrenarlo.

    Si `calibrate=True` en la config, envuelve en CalibratedClassifierCV con el método
    apropiado al algoritmo (isotonic para árboles/ensembles, sigmoid para LR/MLP).

    Para construcción + entrenamiento con fallback robusto ante fallos de calibración,
    usar `fit_with_calibration`.
    """
    base = _build_base_model(model_cfg, random_state)
    if not model_cfg.get("calibrate", False):
        return base

    method = _resolve_calibration_method(
        model_cfg["class"],
        override=model_cfg.get("calibration_method"),
    )
    return _wrap_calibrated(base, method=method)


def fit_with_calibration(
    model_cfg: dict,
    X,
    y,
    random_state: int = 42,
) -> tuple[Any, dict]:
    """
    Construye y entrena el modelo intentando calibrarlo con el método primario;
    si falla, cae a `sigmoid`; si vuelve a fallar, devuelve el modelo sin calibrar
    con un warning explícito.

    Retorna `(fitted_model, calibration_info)` donde calibration_info es:
        {
            "calibrated": bool,
            "method": "isotonic" | "sigmoid" | None,
            "cv": int | None,
            "fallback_reason": str | None,
            "warnings": list[str],
        }

    El modelo final entrenado es lo que debe persistirse con joblib.
    """
    class_name = model_cfg["class"]
    calibrate_requested = bool(model_cfg.get("calibrate", False))

    if not calibrate_requested:
        base = _build_base_model(model_cfg, random_state)
        base.fit(X, y)
        return base, {
            "calibrated": False,
            "method": None,
            "cv": None,
            "fallback_reason": "calibration_disabled_in_config",
            "warnings": [],
        }

    primary_method = _resolve_calibration_method(
        class_name, override=model_cfg.get("calibration_method")
    )
    cv = int(model_cfg.get("calibration_cv", 5))

    methods_to_try = [primary_method] + [
        m for m in CALIBRATION_FALLBACK_ORDER if m != primary_method
    ]

    last_error: Exception | None = None
    warnings_accum: list[str] = []

    for attempt_method in methods_to_try:
        try:
            base = _build_base_model(model_cfg, random_state)
            wrapped = _wrap_calibrated(base, method=attempt_method, cv=cv)
            wrapped.fit(X, y)

            fallback_reason = None
            if attempt_method != primary_method:
                fallback_reason = (
                    f"primary_method_{primary_method}_failed: {type(last_error).__name__}: {last_error}"
                )
                warnings_accum.append(
                    f"calibration_fallback_to_{attempt_method}"
                )
                logger.warning(
                    f"[{class_name}] Calibración con '{primary_method}' falló "
                    f"({type(last_error).__name__}: {last_error}). "
                    f"Cayendo a '{attempt_method}'."
                )

            return wrapped, {
                "calibrated": True,
                "method": attempt_method,
                "cv": cv,
                "fallback_reason": fallback_reason,
                "warnings": warnings_accum,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[{class_name}] Falló calibración con método '{attempt_method}': "
                f"{type(exc).__name__}: {exc}"
            )

    warnings_accum.append("model_not_calibrated")
    logger.error(
        f"[{class_name}] No se pudo calibrar con ningún método. "
        f"Entrenando sin calibración. Último error: {type(last_error).__name__}: {last_error}"
    )
    base = _build_base_model(model_cfg, random_state)
    base.fit(X, y)
    return base, {
        "calibrated": False,
        "method": None,
        "cv": None,
        "fallback_reason": (
            f"all_methods_failed: {type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "all_methods_failed"
        ),
        "warnings": warnings_accum,
    }
