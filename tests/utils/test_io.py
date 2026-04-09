import json
import joblib
import pandas as pd
import pytest
from pathlib import Path
from src.utils.io import read_parquet, write_parquet, read_json, write_json, write_joblib, read_joblib


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def test_write_and_read_parquet(tmp_dir):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_dir / "test.parquet"
    write_parquet(df, path)
    result = read_parquet(path)
    pd.testing.assert_frame_equal(df, result)


def test_write_parquet_creates_parent_dirs(tmp_dir):
    df = pd.DataFrame({"a": [1]})
    path = tmp_dir / "nested" / "dir" / "test.parquet"
    write_parquet(df, path)
    assert path.exists()


def test_write_and_read_json(tmp_dir):
    data = {"key": "value", "num": 42, "list": [1, 2, 3]}
    path = tmp_dir / "test.json"
    write_json(data, path)
    result = read_json(path)
    assert result == data


def test_write_and_read_joblib(tmp_dir):
    obj = {"model": "fake", "params": [1, 2]}
    path = tmp_dir / "model.joblib"
    write_joblib(obj, path)
    result = read_joblib(path)
    assert result == obj


def test_read_parquet_missing_file(tmp_dir):
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_dir / "nonexistent.parquet")
