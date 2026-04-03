from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str
    dataset_version: str
    paths: dict
    train_test_split: dict
    cross_validation: dict
    hyperparameter_search: dict
    active_targets: list[str]
    targets: dict
    cleaning: dict
    features: dict
    models: dict

    def enabled_models(self) -> dict:
        """Retorna solo los modelos con enabled=true."""
        return {
            name: cfg
            for name, cfg in self.models.items()
            if cfg.get("enabled", True)
        }

    def get_target(self, target_name: str) -> dict:
        if target_name not in self.targets:
            raise KeyError(f"Target '{target_name}' no definido en target_config.yaml")
        return self.targets[target_name]

    def raw_data_path(self) -> Path:
        return Path(self.paths["raw_data"])

    def output_path(self) -> Path:
        return Path(self.paths["output"]) / self.pipeline_version


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path | str = "config") -> PipelineConfig:
    config_dir = Path(config_dir)

    pipeline = _load_yaml(config_dir / "pipeline_config.yaml")
    target = _load_yaml(config_dir / "target_config.yaml")
    cleaning = _load_yaml(config_dir / "cleaning_config.yaml")
    features = _load_yaml(config_dir / "features_config.yaml")
    models = _load_yaml(config_dir / "models_config.yaml")

    return PipelineConfig(
        pipeline_version=str(pipeline["pipeline_version"]),
        dataset_version=str(pipeline["dataset_version"]),
        paths=pipeline["paths"],
        train_test_split=pipeline["train_test_split"],
        cross_validation=pipeline["cross_validation"],
        hyperparameter_search=pipeline["hyperparameter_search"],
        active_targets=target["active_targets"],
        targets=target["targets"],
        cleaning=cleaning,
        features=features,
        models=models["models"],
    )
