"""
Model Registry
==============
Scans the models/ directory on startup, loads every .joblib artifact into
memory, and provides CRUD operations for runtime model management.

Expected artifact structure (produced by scripts/train_and_save.py):
{
    "model":               any sklearn-compatible fitted model,
    "preprocessor":        fitted sklearn Pipeline/ColumnTransformer | None,
    "feature_columns":     list[str],          # ordered feature names
    "optimal_threshold":   float,              # F2-optimised threshold
    "training_medians":    dict[str, float],   # median per feature (for imputation)
    "imputation_strategies": {
        "fill_zero":   list[str],
        "fill_median": list[str],
    },
    "version":             str,                # clinical target version name
    "model_name":          str,                # human-readable model name
    "model_type":          str,                # "tree" | "linear" | "ensemble" | "neural"
    "metrics":             dict[str, float],   # validation metrics
    "feature_importance":  list[dict],         # [{"feature": str, "importance": float}]
    "created_at":          str,                # ISO date string
    "description":         str,
    "min_age":             int,                # minimum patient age (default 18)
    "target_description":  str,
}
"""

import logging
from pathlib import Path
from typing import Any

import joblib

from api.core.config import settings
from api.schemas.responses import (
    FeatureImportanceItem,
    ModelInfo,
    ModelMetrics,
)

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {
    "model",
    "feature_columns",
    "optimal_threshold",
    "version",
    "model_name",
    "model_type",
}


def _validate_artifact(artifact: dict, model_id: str) -> None:
    missing = _REQUIRED_KEYS - set(artifact.keys())
    if missing:
        raise ValueError(
            f"Artifact '{model_id}' is missing required keys: {missing}"
        )
    if not isinstance(artifact["feature_columns"], list) or not artifact["feature_columns"]:
        raise ValueError(f"Artifact '{model_id}': 'feature_columns' must be a non-empty list.")
    threshold = artifact["optimal_threshold"]
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"Artifact '{model_id}': 'optimal_threshold' must be in [0, 1], got {threshold}."
        )


def _build_model_info(model_id: str, artifact: dict) -> ModelInfo:
    raw_metrics = artifact.get("metrics", {})
    metrics = ModelMetrics(
        roc_auc=raw_metrics.get("ROC-AUC") or raw_metrics.get("roc_auc"),
        pr_auc=raw_metrics.get("PR-AUC") or raw_metrics.get("pr_auc"),
        recall=raw_metrics.get("Recall") or raw_metrics.get("recall"),
        precision=raw_metrics.get("Precision") or raw_metrics.get("precision"),
        f1=raw_metrics.get("F1") or raw_metrics.get("f1"),
        f2=raw_metrics.get("F2") or raw_metrics.get("f2"),
        accuracy=raw_metrics.get("Accuracy") or raw_metrics.get("accuracy"),
        specificity=raw_metrics.get("Specificity") or raw_metrics.get("specificity"),
        brier_score=raw_metrics.get("Brier") or raw_metrics.get("brier_score"),
        extra={
            k: v
            for k, v in raw_metrics.items()
            if k
            not in {
                "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "F2",
                "Accuracy", "Specificity", "Brier",
                "roc_auc", "pr_auc", "recall", "precision", "f1", "f2",
                "accuracy", "specificity", "brier_score",
            }
        },
    )

    raw_importance = artifact.get("feature_importance", [])
    top_features = [
        FeatureImportanceItem(
            feature=item["feature"] if isinstance(item, dict) else item[0],
            importance=item["importance"] if isinstance(item, dict) else item[1],
        )
        for item in raw_importance[:20]
    ]

    return ModelInfo(
        model_id=model_id,
        model_name=artifact.get("model_name", "Unknown"),
        model_type=artifact.get("model_type", "unknown"),
        version=artifact.get("version", "unknown"),
        description=artifact.get("description", ""),
        n_features=len(artifact["feature_columns"]),
        optimal_threshold=artifact["optimal_threshold"],
        min_age=artifact.get("min_age", 18),
        created_at=artifact.get("created_at", ""),
        metrics=metrics,
        top_features=top_features,
    )


class ModelRegistry:
    """In-memory store of loaded model artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._infos: dict[str, ModelInfo] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_from_directory(self, directory: Path) -> None:
        """Scan *directory* and load every *.joblib file."""
        directory = Path(directory)
        if not directory.exists():
            logger.warning("Models directory '%s' does not exist.", directory)
            return

        for path in sorted(directory.glob("*.joblib")):
            model_id = path.stem
            try:
                self._load_file(path, model_id)
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", model_id, exc)

        logger.info("ModelRegistry: %d model(s) loaded.", len(self._artifacts))

    def _load_file(self, path: Path, model_id: str) -> None:
        artifact: dict = joblib.load(path)
        _validate_artifact(artifact, model_id)
        self._artifacts[model_id] = artifact
        self._infos[model_id] = _build_model_info(model_id, artifact)
        logger.info("  [OK] Loaded '%s' (%s / %s)", model_id, artifact["model_name"], artifact["version"])

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_artifact(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._artifacts:
            raise KeyError(f"Model '{model_id}' not found. Available: {self.list_ids()}")
        return self._artifacts[model_id]

    def get_info(self, model_id: str) -> ModelInfo:
        if model_id not in self._infos:
            raise KeyError(f"Model '{model_id}' not found.")
        return self._infos[model_id]

    def list_ids(self) -> list[str]:
        return sorted(self._artifacts.keys())

    def list_infos(self) -> list[ModelInfo]:
        return [self._infos[mid] for mid in self.list_ids()]

    def __len__(self) -> int:
        return len(self._artifacts)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def register(self, model_id: str, artifact: dict, save_dir: Path | None = None) -> ModelInfo:
        """Validate and register an artifact, optionally persisting it to disk."""
        _validate_artifact(artifact, model_id)
        self._artifacts[model_id] = artifact
        info = _build_model_info(model_id, artifact)
        self._infos[model_id] = info

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            dest = save_dir / f"{model_id}.joblib"
            joblib.dump(artifact, dest)
            logger.info("Saved artifact to '%s'.", dest)

        return info

    def remove(self, model_id: str, models_dir: Path | None = None) -> None:
        """Remove a model from memory and optionally delete the file."""
        if model_id not in self._artifacts:
            raise KeyError(f"Model '{model_id}' not found.")
        del self._artifacts[model_id]
        del self._infos[model_id]

        if models_dir is not None:
            path = Path(models_dir) / f"{model_id}.joblib"
            if path.exists():
                path.unlink()
                logger.info("Deleted artifact file '%s'.", path)


registry = ModelRegistry()


def load_registry() -> None:
    """Call once at application startup."""
    registry.load_from_directory(settings.models_dir)
