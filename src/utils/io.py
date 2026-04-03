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
    df.to_parquet(path, index=False)


def read_json(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_joblib(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def read_joblib(path: Path | str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")
    return joblib.load(path)
