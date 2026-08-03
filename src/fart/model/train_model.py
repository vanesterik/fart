from typing import Tuple

import pandas as pd
import polars as pl

from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.train_test_split import train_test_split
from fart.settings import Settings
from fart.utils import get_candle_filepath


def prepare_training_data(
    settings: Settings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    filepath = get_candle_filepath(settings)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)
    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)
