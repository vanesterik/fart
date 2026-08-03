from pathlib import Path
from typing import Tuple

import polars as pl
from loguru import logger

from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath


def prepare_training_data(
    data_dir: Path,
    market: str,
    interval: str,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    filepath = get_candle_filepath(data_dir, market, interval)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)
    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)


def train(data_dir: Path, market: str, interval: str) -> None:
    X_train, X_test, y_train, y_test = prepare_training_data(data_dir, market, interval)
    logger.info(
        f"Prepared training data: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )
