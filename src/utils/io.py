import json
import joblib
import pandas as pd
from pathlib import Path


def read_parquet(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    # Drop duplicate columns — keep first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    # Cast mixed-type object columns to str so PyArrow doesn't fail on
    # columns like "Inicio anestesia" that contain both strings and ints.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.to_parquet(path, index=False)


def read_json(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def write_joblib(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def read_joblib(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")
    return joblib.load(path)
