import pandas as pd


def calculate_duration(start_series: pd.Series, end_series: pd.Series) -> pd.Series:
    start = pd.to_datetime(start_series, errors="coerce")
    end = pd.to_datetime(end_series, errors="coerce")
    return (end - start).dt.total_seconds() / 60


def print_validation_result(name: str, series: pd.Series, total: int) -> None:
    n_positive = series.sum()
    print(f"  {name}: {n_positive:,} casos ({n_positive / total * 100:.2f}%)")
