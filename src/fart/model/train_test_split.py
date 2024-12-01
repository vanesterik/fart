from typing import Tuple

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split as base_train_test_split

from fart.constants import feature_names as fn


def train_test_split(
    df: pl.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    - df (pl.DataFrame): A DataFrame containing the data to split.
    - test_size (float): The proportion of the data to include in the test
      split. This can also be an integer representing the number of rows to
      include in the test split.

    Returns
    -------
    - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple
      containing the following:
        - X_train (pd.DataFrame): Training data
        - X_test (pd.DataFrame): Testing data
        - y_train (pd.Series): Training target
        - y_test (pd.Series): Testing target

    """

    # Transform to pandas dataframe and drop nulls
    pandas_df = df.to_pandas()

    # Filter predictor columns and reshape to 1D array
    X = pandas_df.drop(columns=fn.TRADE_SIGNAL)

    # Filter target column
    y = pandas_df[fn.TRADE_SIGNAL]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = base_train_test_split(
        X,
        y,
        random_state=42,  # Set random state for reproducibility
        shuffle=False,  # Do not shuffle the data as is time series data
        test_size=test_size,
    )

    # Return the split data in a tuple
    return X_train, X_test, y_train, y_test
