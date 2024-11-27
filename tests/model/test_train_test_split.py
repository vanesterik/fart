import pandas as pd
import polars as pl

from fart.constants import feature_names as fn
from fart.model.train_test_split import train_test_split


def test_train_test_split() -> None:
    # Create a sample DataFrame
    data = {
        fn.BBANDS_UPPER: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.BBANDS_MIDDLE: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.BBANDS_LOWER: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.EMA_FAST: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.EMA_SLOW: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD_SIGNAL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD_HISTOGRAM: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.RSI: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.TIMESTAMP: pd.date_range(start="1/1/2022", periods=10, freq="min"),
        fn.CLOSE: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pl.DataFrame(data)

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(df)

    # Check the shapes of the splits
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2

    # Check the content of the splits
    assert X_train.iloc[0][fn.BBANDS_UPPER] == 1
    assert X_test.iloc[0][fn.BBANDS_UPPER] == 9
    assert y_train.iloc[0] == 1
    assert y_test.iloc[0] == 9


def test_train_test_split_no_nan() -> None:
    # Create a sample DataFrame with NaN values
    data = {
        fn.BBANDS_UPPER: [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        fn.BBANDS_MIDDLE: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.BBANDS_LOWER: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.EMA_FAST: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.EMA_SLOW: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD_SIGNAL: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.MACD_HISTOGRAM: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.RSI: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fn.TIMESTAMP: pd.date_range(start="1/1/2022", periods=10, freq="min"),
        fn.CLOSE: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pl.DataFrame(data)

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(df)

    # Check the shapes of the splits
    assert X_train.shape[0] == 7
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 7
    assert y_test.shape[0] == 2

    # Check the content of the splits
    assert X_train.iloc[0][fn.BBANDS_UPPER] == 1
    assert X_test.iloc[0][fn.BBANDS_UPPER] == 9
    assert y_train.iloc[0] == 1
    assert y_test.iloc[0] == 9
