from __future__ import annotations

import importlib

from api.core.logging import get_logger

logger = get_logger("sklearn_compat")


def _patch_class_tags(cls) -> bool:
    if getattr(cls, "_sklearn_compat_patched", False):
        return False

    original_fn = getattr(cls, "__sklearn_tags__", None)
    if original_fn is None:
        return False

    def __sklearn_tags__(self):
        tags = original_fn(self)
        tags.estimator_type = "classifier"
        return tags

    cls.__sklearn_tags__ = __sklearn_tags__
    cls._sklearn_compat_patched = True
    return True


def apply_sklearn_compat_patches() -> None:
    """Idempotente: parchea XGBClassifier y LGBMClassifier si están disponibles."""
    try:
        from sklearn.base import is_classifier
    except ImportError:
        logger.warning("sklearn no instalado — no se aplican patches.")
        return

    for pkg, cls_name in [("xgboost", "XGBClassifier"), ("lightgbm", "LGBMClassifier")]:
        try:
            mod = importlib.import_module(pkg)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            try:
                is_ok = is_classifier(cls())
            except Exception:
                is_ok = False
            if not is_ok and _patch_class_tags(cls):
                logger.info(f"sklearn compat patch aplicado a {pkg}.{cls_name}")
        except Exception as exc:
            logger.warning(f"No se pudo parchear {pkg}.{cls_name}: {exc}")
