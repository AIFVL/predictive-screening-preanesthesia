# tests/utils/test_config.py
import pytest
import yaml
from pathlib import Path
from src.utils.config import load_config, PipelineConfig


@pytest.fixture
def config_dir(tmp_path):
    """Crea un conjunto mínimo de YAMLs de config para tests."""
    pipeline = {
        "pipeline_version": "test_v1",
        "dataset_version": "v1",
        "paths": {"raw_data": "data_raw/", "output": "output/"},
        "train_test_split": {"test_size": 0.2, "random_state": 42},
        "cross_validation": {"n_folds": 5, "n_jobs": 1},
        "hyperparameter_search": {"enabled": False, "n_iter": 10, "scoring": "f2", "cv_folds": 5},
    }
    target = {
        "active_targets": ["target_a"],
        "targets": {
            "target_a": {
                "threshold": 1,
                "apply_cancel_non_medico_rule": True,
                "description": "Test target",
                "flags": ["flag_cancelacion"],
            }
        },
    }
    cleaning = {
        "identifier_columns": ["CODIGO"],
        "leakage_columns": [],
        "age_filter": {"column": "Edad", "min": 18},
        "outlier_rules": {},
        "encoding_fixes": {},
        "external_services": {},
    }
    features = {
        "numerical_features": ["Edad", "Peso"],
        "features_to_exclude": [],
        "feature_pruning": {"enabled": True, "min_variance": 0.01},
        "encoding_fix_map": {},
    }
    models = {
        "models": {
            "logistic_regression": {
                "enabled": True,
                "module": "sklearn.linear_model",
                "class": "LogisticRegression",
                "target_metric": "f2",
                "params": {"C": 1.0},
                "search_space": {"C": [0.1, 1.0]},
            }
        }
    }

    for name, content in [
        ("pipeline_config.yaml", pipeline),
        ("target_config.yaml", target),
        ("cleaning_config.yaml", cleaning),
        ("features_config.yaml", features),
        ("models_config.yaml", models),
    ]:
        (tmp_path / name).write_text(yaml.dump(content, allow_unicode=True))

    return tmp_path


def test_load_config_returns_pipeline_config(config_dir):
    cfg = load_config(config_dir)
    assert isinstance(cfg, PipelineConfig)


def test_pipeline_version(config_dir):
    cfg = load_config(config_dir)
    assert cfg.pipeline_version == "test_v1"


def test_active_targets(config_dir):
    cfg = load_config(config_dir)
    assert cfg.active_targets == ["target_a"]


def test_target_flags(config_dir):
    cfg = load_config(config_dir)
    assert cfg.targets["target_a"]["flags"] == ["flag_cancelacion"]


def test_enabled_models(config_dir):
    cfg = load_config(config_dir)
    enabled = cfg.enabled_models()
    assert "logistic_regression" in enabled


def test_missing_config_file_raises(tmp_path):
    # Solo pipeline_config.yaml, faltan los demás
    (tmp_path / "pipeline_config.yaml").write_text("pipeline_version: v1")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)
