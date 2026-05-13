from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelManifest:
    """Representación inmutable de un manifest cargado desde disco."""
    schema_version: str
    model_id: str
    target_version: str
    algorithm: str
    model_path: Path           # absoluto, listo para joblib.load
    metrics_path: Path         # absoluto
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    feature_medians: dict[str, float | None]
    imputation: dict[str, Any]
    threshold: float
    threshold_metric: str
    calibrated: bool
    calibration: dict[str, Any]
    prevalence: dict[str, Any]
    performance: dict[str, float | None]
    warnings: list[str]
    created_at: str | None
    raw: dict[str, Any]        # JSON original por si se necesita debug


def load_manifest(manifest_path: Path) -> ModelManifest:
    """Lee un `<algorithm>_manifest.json` y resuelve paths relativos a su carpeta."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_dir = manifest_path.parent
    model_filename = data["model_filename"]
    metrics_filename = data["metrics_filename"]

    return ModelManifest(
        schema_version=data.get("schema_version", "1.0"),
        model_id=data["model_id"],
        target_version=data["target_version"],
        algorithm=data["algorithm"],
        model_path=(base_dir / model_filename).resolve(),
        metrics_path=(base_dir / metrics_filename).resolve(),
        feature_names=list(data["feature_names"]),
        feature_dtypes=dict(data["feature_dtypes"]),
        feature_medians=dict(data["feature_medians"]),
        imputation=dict(data.get("imputation", {"strategy": "fill_constant", "value": -1})),
        threshold=float(data["threshold"]),
        threshold_metric=data.get("threshold_metric", "f2"),
        calibrated=bool(data.get("calibrated", False)),
        calibration=dict(data.get("calibration", {})),
        prevalence=dict(data.get("prevalence", {})),
        performance=dict(data.get("performance", {})),
        warnings=list(data.get("warnings", [])),
        created_at=data.get("created_at"),
        raw=data,
    )
