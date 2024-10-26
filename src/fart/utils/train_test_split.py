from typing import Tuple

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split as base_train_test_split

from fart.constants import feature_names as fn


def train_test_split(
    df: pl.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    - df (pl.DataFrame): A DataFrame containing the data to split.

    Returns
    -------
    - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the following:
        - X_train (pd.DataFrame): Training data
        - X_test (pd.DataFrame): Testing data
        - y_train (pd.Series): Training target
        - y_test (pd.Series): Testing target

    """

    # Transform to pandas dataframe and drop nulls
    pandas_df = df.to_pandas()

    # Drop NaN values
    pandas_df.dropna(inplace=True)

    # Filter predictor columns and reshape to 1D array
    X = pandas_df[
        [
            fn.BBANDS_UPPER,
            fn.BBANDS_MIDDLE,
            fn.BBANDS_LOWER,
            fn.EMA_FAST,
            fn.EMA_SLOW,
            fn.MACD,
            fn.MACD_SIGNAL,
            fn.MACD_HISTOGRAM,
            fn.RSI,
            fn.TIMESTAMP,
        ]
    ]

    # Filter target column
    y = pandas_df[fn.CLOSE]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = base_train_test_split(
        X,
        y,
        random_state=42,  # Set random state for reproducibility
        shuffle=False,  # Do not shuffle the data as is time series data
        test_size=120,  # 120 minutes = 2 hours
    )

    # Return the split data in a tuple
    return X_train, X_test, y_train, y_test
