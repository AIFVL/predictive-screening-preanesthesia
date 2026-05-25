from __future__ import annotations

"""Servicio de explicabilidad por SHAP.

Soporta tres estrategias según el tipo de estimador subyacente:
- TreeExplainer  → modelos basados en árboles (XGBoost, RandomForest, ExtraTrees,
                    HistGradientBoosting, LightGBM, Stacking con árbol base)
- LinearExplainer → regresión logística / modelos lineales
- KernelExplainer → fallback genérico (MLP, etc.)

El resultado es una lista ordenada (de mayor a menor |shap|) de
  { feature, value, shap_value }
limitada a `top_n` contribuciones.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from api.domain.manifest import ModelManifest
from api.domain.registry import ModelRegistry
from api.services.preprocessor import preprocess

# Modelos que usan TreeExplainer
_TREE_ALGORITHMS = {
    "xgboost",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
}

# Modelos lineales
_LINEAR_ALGORITHMS = {"logistic_regression"}


@dataclass(frozen=True)
class ShapContribution:
    feature: str
    value: float | None   # valor original del paciente (antes de imputar)
    shap_value: float     # contribución SHAP (puede ser + o -)


def _unwrap_calibrated(model: Any) -> Any:
    """Si el modelo es un CalibratedClassifierCV, devuelve el estimador base."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model



def _safe_predict_proba(model: Any):
    """Devuelve una función callable pura (no un método bound) para que
    KernelExplainer no intente acceder ni mutar atributos del modelo."""
    def _predict(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)
    return _predict


def _choose_explainer(base_model: Any, df: pd.DataFrame, algorithm: str):
    """Construye y devuelve el explainer SHAP apropiado."""
    import shap  # import late para no romper si shap no está instalado

    alg = algorithm.lower()
    # Numpy array: evita conflictos con feature_names_in_ en XGBoost/sklearn
    arr = df.values

    if alg in _TREE_ALGORITHMS:
        try:
            return shap.TreeExplainer(
                base_model,
                feature_perturbation="tree_path_dependent",
            )
        except Exception as _tree_err:
            import logging
            logging.getLogger("api.explainer").warning(
                f"TreeExplainer falló para {alg}: {_tree_err!r} — usando fallback"
            )

    if alg in _LINEAR_ALGORITHMS:
        try:
            masker = shap.maskers.Independent(arr, max_samples=min(100, len(arr)))
            return shap.LinearExplainer(base_model, masker)
        except Exception:
            pass  # fallback

    # Fallback: KernelExplainer con función pura para no exponer el objeto modelo
    background = shap.sample(arr, min(50, len(arr)))
    return shap.KernelExplainer(_safe_predict_proba(base_model), background)


def _extract_shap_vector(shap_values: Any, n_features: int) -> np.ndarray:
    """Normaliza la salida de cualquier explainer a un vector 1-D de shape (n_features,).

    Casos que maneja:
    - list[ndarray]  → shap_values[1] es la clase positiva  (TreeExplainer clásico)
    - ndarray 3-D    → shape (1, n_features, 2), tomar [:, :, 1].squeeze()
    - ndarray 2-D    → shape (1, n_features), tomar [0]
    - ndarray 1-D    → shape (n_features,), usar directamente
    """
    # Lista de arrays por clase (TreeExplainer multiclass / binario antiguo)
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1])          # clase positiva
    else:
        arr = np.asarray(shap_values)

    # 3-D: (n_samples, n_features, n_classes)  — TreeExplainer moderno con check_additivity
    if arr.ndim == 3:
        arr = arr[:, :, 1]                        # clase positiva → (n_samples, n_features)

    # 2-D: (n_samples, n_features) — quitar dimensión de batch
    if arr.ndim == 2:
        arr = arr[0]                              # → (n_features,)

    # Sanity check
    if arr.ndim != 1 or arr.shape[0] != n_features:
        raise ValueError(
            f"Forma SHAP inesperada tras aplanado: {arr.shape}. "
            f"Se esperaba ({n_features},)."
        )

    return arr


def explain_one(
    features: dict,
    manifest: ModelManifest,
    registry: ModelRegistry,
    top_n: int = 10,
) -> list[ShapContribution]:
    """
    Calcula valores SHAP para una única observación.

    Parameters
    ----------
    features : dict
        Valores crudos del paciente (pueden tener None).
    manifest : ModelManifest
        Manifest del modelo.
    registry : ModelRegistry
        Registry para cargar el modelo.
    top_n : int
        Número de contribuciones principales a devolver.

    Returns
    -------
    list[ShapContribution]
        Lista de hasta `top_n` contribuciones ordenadas por |shap_value| desc.
    """
    import shap  # noqa: F401

    # Recuperar slug para cargar el modelo
    target_slug = _slug_from_manifest(manifest, registry)
    model = registry.load_model(target_slug=target_slug, algorithm=manifest.algorithm)
    base_model = _unwrap_calibrated(model)

    # Preprocesar como siempre
    df = preprocess(features, manifest)
    # Numpy array: evita que SHAP/XGBoost intente sobreescribir feature_names_in_
    arr = df.values

    # XGBoost calibrado: TreeExplainer falla por bug con valores de split en
    # notación científica con corchetes dentro del JSON del árbol.
    # shap.Explainer (API unificada ≥ 0.40) lo maneja correctamente pasando
    # el modelo calibrado completo, que expone predict_proba sin ese parser.
    # Stacking: igual, no es un árbol único → KernelExplainer sobre modelo completo.
    if manifest.algorithm in ("xgboost", "stacking"):
        # KernelExplainer con 1 punto de background da ceros (sin varianza).
        # PermutationExplainer no necesita background con varianza: evalúa el
        # impacto permutando cada feature respecto a la fila de referencia.
        background = pd.DataFrame(
            [manifest.feature_medians],
            columns=manifest.feature_names,
        ).values
        masker = shap.maskers.Independent(background, max_samples=1)
        explainer = shap.PermutationExplainer(
            _safe_predict_proba(model),
            masker,
        )
        # shap_output shape: (1, n_features, n_classes) — tomamos clase 1
        shap_output = explainer(arr)
        shap_values = shap_output.values  # ndarray (1, n_features, n_classes)
        import logging as _l
        _sv = np.asarray(shap_values)
        _l.getLogger("api.explainer").warning(
            f"[SHAP DEBUG] PermutationExplainer values shape={_sv.shape}, "
            f"nonzero={np.count_nonzero(_sv)}, "
            f"flat[:5]={list(_sv.flat[:5])}, "
            f"base_values={shap_output.base_values}, "
            f"predict_on_input={_safe_predict_proba(model)(arr)}"
        )
    else:
        explainer = _choose_explainer(base_model, df, manifest.algorithm)
        try:
            shap_values = explainer.shap_values(arr, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(arr)

    sv = _extract_shap_vector(shap_values, n_features=len(manifest.feature_names))

    feature_names = manifest.feature_names

    contributions = []
    for i, fname in enumerate(feature_names):
        raw_val = features.get(fname)  # valor original (puede ser None)
        contributions.append(
            ShapContribution(
                feature=fname,
                value=float(raw_val) if raw_val is not None else None,
                shap_value=float(sv[i]),
            )
        )

    # Ordenar por magnitud descendente, tomar top_n
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
    return contributions[:top_n]


def _slug_from_manifest(manifest: ModelManifest, registry: ModelRegistry) -> str:
    for alias in registry._settings.target_aliases:  # noqa: SLF001
        if alias.target_version == manifest.target_version:
            return alias.slug
    raise KeyError(f"No hay alias para target_version={manifest.target_version!r}")
