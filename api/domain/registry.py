from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import joblib

from api.core.config import Settings, TargetAlias
from api.core.logging import get_logger
from api.domain.manifest import ModelManifest, load_manifest

logger = get_logger("registry")


@dataclass(frozen=True)
class RegistryKey:
    """Identificador interno: (slug_publico, algoritmo)."""
    target_slug: str
    algorithm: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.target_slug, self.algorithm)


class ModelRegistry:

    def __init__(self, settings: Settings):
        self._settings = settings
        self._manifests: dict[RegistryKey, ModelManifest] = {}
        self._aliases_by_slug: dict[str, TargetAlias] = {a.slug: a for a in settings.target_aliases}
        self._aliases_by_target_version: dict[str, TargetAlias] = {
            a.target_version: a for a in settings.target_aliases
        }
        self._model_cache: OrderedDict[RegistryKey, object] = OrderedDict()
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------- bootstrap

    def discover(self) -> None:
        """Recorre `settings.models_dir` y registra manifests de targets en alcance."""
        root = self._settings.models_dir
        if not root.is_dir():
            logger.warning(f"models_dir no existe: {root}")
            return

        for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            target_version = target_dir.name
            alias = self._aliases_by_target_version.get(target_version)
            if alias is None:
                logger.info(f"target_version='{target_version}' fuera de alcance — saltando.")
                continue

            for manifest_file in sorted(target_dir.glob("*_manifest.json")):
                try:
                    manifest = load_manifest(manifest_file)
                except Exception as exc:
                    logger.error(f"Manifest inválido en {manifest_file}: {exc}")
                    continue

                if manifest.target_version != target_version:
                    logger.warning(
                        f"Manifest {manifest_file} declara target_version="
                        f"'{manifest.target_version}' pero está en carpeta "
                        f"'{target_version}'. Saltando."
                    )
                    continue

                if not manifest.model_path.exists():
                    logger.warning(
                        f"Manifest {manifest_file} apunta a un modelo inexistente: "
                        f"{manifest.model_path}. Saltando."
                    )
                    continue

                key = RegistryKey(target_slug=alias.slug, algorithm=manifest.algorithm)
                self._manifests[key] = manifest
                logger.info(
                    f"  registrado: {alias.slug}/{manifest.algorithm} "
                    f"(calibrated={manifest.calibrated}, "
                    f"method={manifest.calibration.get('method')})"
                )

        logger.info(f"ModelRegistry listo: {len(self._manifests)} modelos en alcance.")

    # ----------------------------------------------------------------- catálogos

    def list_targets(self) -> list[dict]:
        """Para `GET /targets`. Cuenta modelos por slug."""
        counts: dict[str, int] = {}
        for key in self._manifests:
            counts[key.target_slug] = counts.get(key.target_slug, 0) + 1
        result = []
        for alias in self._settings.target_aliases:
            if counts.get(alias.slug, 0) == 0:
                continue
            result.append({
                "slug": alias.slug,
                "display_name": alias.display_name,
                "description": alias.description,
                "recommended": alias.recommended,
                "n_models": counts.get(alias.slug, 0),
            })
        return result

    def list_models(self) -> list[dict]:
        """Para `GET /models`. Vista plana ligera de todos los modelos en alcance."""
        result = []
        for key, manifest in self._manifests.items():
            alias = self._aliases_by_slug[key.target_slug]
            result.append({
                "target": key.target_slug,
                "target_display_name": alias.display_name,
                "algorithm": key.algorithm,
                "model_id": manifest.model_id,
                "calibrated": manifest.calibrated,
                "calibration_method": manifest.calibration.get("method"),
                "threshold": manifest.threshold,
                "performance": manifest.performance,
                "warnings": manifest.warnings,
                "recommended": (
                    alias.recommended
                    and key.algorithm in ("xgboost", "stacking")
                ),
            })
        return result

    def get_manifest(self, target_slug: str, algorithm: str) -> ModelManifest | None:
        return self._manifests.get(RegistryKey(target_slug=target_slug, algorithm=algorithm))

    def has(self, target_slug: str, algorithm: str) -> bool:
        return RegistryKey(target_slug=target_slug, algorithm=algorithm) in self._manifests

    def alias(self, target_slug: str) -> TargetAlias | None:
        return self._aliases_by_slug.get(target_slug)

    # --------------------------------------------------------------- LRU loading

    def load_model(self, target_slug: str, algorithm: str):
        """Devuelve el estimador `.joblib` cargado en memoria. Lazy + LRU."""
        key = RegistryKey(target_slug=target_slug, algorithm=algorithm)
        manifest = self._manifests.get(key)
        if manifest is None:
            raise KeyError(f"Modelo no registrado: {target_slug}/{algorithm}")

        with self._cache_lock:
            cached = self._model_cache.get(key)
            if cached is not None:
                self._model_cache.move_to_end(key)
                return cached

        logger.info(f"Cargando modelo {key.as_tuple()} desde {manifest.model_path}")
        model = joblib.load(manifest.model_path)

        with self._cache_lock:
            self._model_cache[key] = model
            self._model_cache.move_to_end(key)
            while len(self._model_cache) > self._settings.model_cache_size:
                evicted_key, _ = self._model_cache.popitem(last=False)
                logger.info(f"Caché LRU desalojó: {evicted_key.as_tuple()}")
        return model

    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._model_cache)

    def n_registered(self) -> int:
        return len(self._manifests)
