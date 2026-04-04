from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.io import write_parquet
from src.utils.logger import get_logger

logger = get_logger("datasets.splits")


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split estratificado train/test.
    Si out_dir se especifica, escribe los 4 parquets en esa carpeta.
    Retorna (X_train, X_test, y_train, y_test).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size debe estar entre 0 y 1, recibido: {test_size}")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split: {len(X_train):,} train / {len(X_test):,} test "
        f"(positivos train={y_train.mean() * 100:.1f}%, test={y_test.mean() * 100:.1f}%)"
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        write_parquet(X_train, out_dir / "X_train.parquet")
        write_parquet(X_test, out_dir / "X_test.parquet")
        write_parquet(pd.DataFrame({"target": y_train}), out_dir / "y_train.parquet")
        write_parquet(pd.DataFrame({"target": y_test}), out_dir / "y_test.parquet")
        logger.info(f"Splits guardados en {out_dir}")

    return X_train, X_test, y_train, y_test
