from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from api.domain.manifest import ModelManifest
from api.domain.registry import ModelRegistry
from api.services.preprocessor import preprocess


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    probability: float
    threshold: float
    risk_level: str
    calibrated: bool
    warnings: list[str]


def _risk_level(probability: float, threshold: float) -> str:
    """Categoriza el nivel de riesgo respecto al threshold óptimo del modelo."""
    if probability >= threshold * 1.5:
        return "high"
    if probability >= threshold:
        return "elevated"
    if probability >= threshold * 0.5:
        return "moderate"
    return "low"


def predict_one(
    features: dict,
    manifest: ModelManifest,
    registry: ModelRegistry,
) -> PredictionResult:
    df = preprocess(features, manifest)
    model = registry.load_model(target_slug=_slug_from_manifest(manifest, registry),
                                algorithm=manifest.algorithm)
    proba = float(model.predict_proba(df)[0, 1])
    pred = int(proba >= manifest.threshold)

    warnings = list(manifest.warnings)
    if not manifest.calibrated:
        warnings.append("model_not_calibrated")

    return PredictionResult(
        predicted_class=pred,
        probability=proba,
        threshold=manifest.threshold,
        risk_level=_risk_level(proba, manifest.threshold),
        calibrated=manifest.calibrated,
        warnings=_dedupe_preserving_order(warnings),
    )


def predict_batch(
    features_list: list[dict],
    manifest: ModelManifest,
    registry: ModelRegistry,
) -> list[PredictionResult]:
    df = preprocess(features_list, manifest)
    model = registry.load_model(target_slug=_slug_from_manifest(manifest, registry),
                                algorithm=manifest.algorithm)
    probas = np.asarray(model.predict_proba(df))[:, 1]
    preds = (probas >= manifest.threshold).astype(int)

    base_warnings = list(manifest.warnings)
    if not manifest.calibrated:
        base_warnings.append("model_not_calibrated")
    base_warnings = _dedupe_preserving_order(base_warnings)

    return [
        PredictionResult(
            predicted_class=int(preds[i]),
            probability=float(probas[i]),
            threshold=manifest.threshold,
            risk_level=_risk_level(float(probas[i]), manifest.threshold),
            calibrated=manifest.calibrated,
            warnings=base_warnings,
        )
        for i in range(len(probas))
    ]


def _slug_from_manifest(manifest: ModelManifest, registry: ModelRegistry) -> str:
    """Recupera el slug público a partir del target_version del manifest."""
    for alias in registry._settings.target_aliases:  # noqa: SLF001 — uso interno controlado
        if alias.target_version == manifest.target_version:
            return alias.slug
    raise KeyError(f"No hay alias para target_version={manifest.target_version!r}")


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
