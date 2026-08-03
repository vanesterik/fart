import math
from typing import Tuple

import polars as pl

from fart.constants import CLOSE


def train_test_split(
    df: pl.DataFrame,
    target: str = CLOSE,
    test_size: float = 0.2,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    - df (pl.DataFrame): A DataFrame containing the data to split.
    - target (str): The name of the target column.
    - test_size (float): The proportion of the data to include in the test
      split.

    Returns
    -------
    - Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]: A tuple
      containing the following:
        - X_train (pl.DataFrame): Training data
        - X_test (pl.DataFrame): Testing data
        - y_train (pl.Series): Training target
        - y_test (pl.Series): Testing target

    """

    # Split by row order, not shuffled, since this is time series data
    n_test = math.ceil(df.height * test_size)
    n_train = df.height - n_test

    train_df = df.head(n_train)
    test_df = df.tail(n_test)

    # Filter predictor columns and target column
    X_train = train_df.drop(target)
    y_train = train_df[target]
    X_test = test_df.drop(target)
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test
