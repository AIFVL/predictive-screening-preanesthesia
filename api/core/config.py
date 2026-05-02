from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TargetAlias:
    slug: str
    target_version: str
    display_name: str
    description: str
    recommended: bool


TARGET_ALIASES: list[TargetAlias] = [
    TargetAlias(
        slug="general_risk",
        target_version="target_d_v2_hosp",
        display_name="Riesgo general",
        description=(
            "Target amplio que cubre múltiples flags clínicos preoperatorios "
            "(no solo hospitalización)."
        ),
        recommended=False,
    ),
    TargetAlias(
        slug="hospitalization_risk",
        target_version="target_f_predictibilidad_maxima",
        display_name="Riesgo de hospitalización / UCI",
        description=(
            "Target específico de outcomes graves: hospitalización prolongada, "
            "UCI y eventos adversos críticos. Mayor predictibilidad."
        ),
        recommended=True,
    ),
]


def _slug_to_alias() -> dict[str, TargetAlias]:
    return {a.slug: a for a in TARGET_ALIASES}


def _target_version_to_alias() -> dict[str, TargetAlias]:
    return {a.target_version: a for a in TARGET_ALIASES}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    models_dir: Path
    cors_origins: list[str]
    log_level: str
    api_title: str = "Preanesthesia Screening API"
    api_version: str = "1.0.0"
    max_batch_size: int = 100
    model_cache_size: int = 8
    target_aliases: list[TargetAlias] = field(default_factory=lambda: list(TARGET_ALIASES))

    def alias_by_slug(self, slug: str) -> TargetAlias | None:
        return _slug_to_alias().get(slug)

    def alias_by_target_version(self, target_version: str) -> TargetAlias | None:
        return _target_version_to_alias().get(target_version)


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


def get_settings() -> Settings:
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    models_dir = Path(os.environ.get("MODELS_DIR", project_root / "output" / "v2" / "models")).resolve()
    return Settings(
        project_root=project_root,
        models_dir=models_dir,
        cors_origins=_parse_csv_env("CORS_ORIGINS", "*"),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "100")),
        model_cache_size=int(os.environ.get("MODEL_CACHE_SIZE", "8")),
    )
