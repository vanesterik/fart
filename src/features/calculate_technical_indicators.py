# External imports
import pandas as pd
from talib import BBANDS, EMA, MACD, RSI

# Internal imports
from constants import column_names as cn


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame.

    Parameters
    ----------
    - data_frame (pd.DataFrame): A DataFrame containing close price data.

    Returns
    -------
    - pd.DataFrame: A DataFrame containing the data with the calculated indicators:
        - Bollinger Bands (Upper, Middle, Lower)
        - Exponential Moving Averages (6, 9, 12)
        - Moving Average Convergence Divergence (MACD, Signal, Histogram)
        - Relative Strength Index

    Example
    -------
    >>> df = pd.DataFrame({
    ...     "close": [1, 2, 3, 4, 5],
    ... })
    >>> build_features(df)
    Close  ...  Relative Strength Index
    0      ...  100.000000
    1      ...   50.000000

    """
    # Bollinger Bands
    df[cn.BBANDS_UPPER], df[cn.BBANDS_MIDDLE], df[cn.BBANDS_LOWER] = BBANDS(
        df[cn.CLOSE],
        timeperiod=10,
        nbdevdn=1.5,
        nbdevup=1.5,
    )

    # Exponential Moving Averages
    df[cn.EMA_6] = EMA(df[cn.CLOSE], timeperiod=6)
    df[cn.EMA_9] = EMA(df[cn.CLOSE], timeperiod=9)
    df[cn.EMA_12] = EMA(df[cn.CLOSE], timeperiod=12)

    # Moving Average Convergence Divergence
    df[cn.MACD], df[cn.MACD_SIGNAL], df[cn.MACD_HISTOGRAM] = MACD(
        df[cn.CLOSE],
        fastperiod=6,
        slowperiod=13,
        signalperiod=5,
    )

    # Relative Strength Index
    df[cn.RSI] = RSI(df[cn.CLOSE], timeperiod=5)

    # Remove rows with NaN values due to calculations of indicators
    df.dropna(inplace=True)

    return df
