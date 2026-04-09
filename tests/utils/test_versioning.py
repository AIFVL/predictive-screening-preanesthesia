import pytest
import yaml
import json
from pathlib import Path
from src.utils.versioning import compute_pipeline_hash, resolve_version_dir, is_version_cached


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "pipeline_config.yaml").write_text("pipeline_version: v1\n")
    (tmp_path / "target_config.yaml").write_text("active_targets: [target_a]\n")
    (tmp_path / "cleaning_config.yaml").write_text("identifier_columns: [CODIGO]\n")
    (tmp_path / "features_config.yaml").write_text("numerical_features: [Edad]\n")
    (tmp_path / "models_config.yaml").write_text("models: {}\n")
    return tmp_path


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "OPERA_PRE.xlsx").write_bytes(b"fake_pre_data")
    (tmp_path / "OPERA_POS.xlsx").write_bytes(b"fake_pos_data")
    return tmp_path


def test_compute_hash_is_deterministic(config_dir, data_dir):
    h1 = compute_pipeline_hash(config_dir, data_dir)
    h2 = compute_pipeline_hash(config_dir, data_dir)
    assert h1 == h2


def test_compute_hash_changes_on_config_change(config_dir, data_dir):
    h1 = compute_pipeline_hash(config_dir, data_dir)
    (config_dir / "pipeline_config.yaml").write_text("pipeline_version: v2\n")
    h2 = compute_pipeline_hash(config_dir, data_dir)
    assert h1 != h2


def test_compute_hash_changes_on_data_change(config_dir, data_dir):
    h1 = compute_pipeline_hash(config_dir, data_dir)
    (data_dir / "OPERA_PRE.xlsx").write_bytes(b"different_data")
    h2 = compute_pipeline_hash(config_dir, data_dir)
    assert h1 != h2


def test_resolve_version_dir(tmp_path):
    version_dir = resolve_version_dir(tmp_path / "output", "v1")
    assert version_dir == tmp_path / "output" / "v1"


def test_is_version_cached_false_when_no_hash_file(tmp_path, config_dir, data_dir):
    output_dir = tmp_path / "output" / "v1"
    output_dir.mkdir(parents=True)
    assert not is_version_cached(output_dir, config_dir, data_dir)


def test_is_version_cached_true_when_hash_matches(tmp_path, config_dir, data_dir):
    output_dir = tmp_path / "output" / "v1"
    output_dir.mkdir(parents=True)
    h = compute_pipeline_hash(config_dir, data_dir)
    (output_dir / "pipeline_hash.json").write_text(json.dumps({"hash": h}))
    assert is_version_cached(output_dir, config_dir, data_dir)
