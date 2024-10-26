# External imports
import polars as pl
from talib import BBANDS, EMA, MACD, RSI

# Internal imports
from fart.constants import feature_names as fn
from fart.features.technical_indicators_config import TechnicalIndicatorsConfig


def calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate technical indicators for a given DataFrame.

    Parameters
    ----------
    - data_frame (pl.DataFrame): A DataFrame containing close price data.

    Returns
    -------
    - pl.DataFrame: A DataFrame containing the data with the calculated
      indicators:
        - Bollinger Bands (Upper, Middle, Lower):
            - Period: 20
            - Standard deviation: 2
        - Exponential Moving Averages:
            - Fast period: 9 periods
            - Slow period: 21 periods
        - Moving Average Convergence Divergence (MACD, Signal, Histogram):
            - Fast period: 12 periods
            - Slow period: 26 periods
            - Signal line: 9-period EMA of the MACD line
        - Relative Strength Index:
            - Period: 14
            - Overbought level: 70
            - Oversold level: 30

    Example
    -------
    >>> df = pl.DataFrame({
    ...     "close": [1, 2, 3, 4, 5],
    ... })
    >>> calculate_technical_indicators(df)
    ┌───────┬───────────────────────────────┐
    │ Close ┆ ... ┆ Relative Strength Index │
    ╞═══════╪═════╪═════════════════════════╡
    │ 0     ┆ ... ┆ 100.000000              │
    │ 1     ┆ ... ┆  50.000000              │
    └───────┴─────┴─────────────────────────┘

    """
    config = TechnicalIndicatorsConfig()

    # Bollinger Bands
    bbands_upper, bbands_middle, bbands_lower = BBANDS(
        df[fn.CLOSE],
        timeperiod=config.bbands.period,
        nbdevdn=config.bbands.standard_deviation,
        nbdevup=config.bbands.standard_deviation,
    )

    # Exponential Moving Averages
    ema_fast = EMA(df[fn.CLOSE], timeperiod=config.ema.fast_period)
    ema_slow = EMA(df[fn.CLOSE], timeperiod=config.ema.slow_period)

    # Moving Average Convergence Divergence
    macd, macd_signal, macd_histogram = MACD(
        df[fn.CLOSE],
        fastperiod=config.macd.fast_period,
        slowperiod=config.macd.slow_period,
        signalperiod=config.macd.signal_period,
    )

    # Relative Strength Index
    rsi = RSI(df[fn.CLOSE], timeperiod=config.rsi.period)

    # Return DataFrame with calculated indicators
    return df.with_columns(
        [
            pl.Series(fn.BBANDS_UPPER, bbands_upper),
            pl.Series(fn.BBANDS_MIDDLE, bbands_middle),
            pl.Series(fn.BBANDS_LOWER, bbands_lower),
            pl.Series(fn.EMA_FAST, ema_fast),
            pl.Series(fn.EMA_SLOW, ema_slow),
            pl.Series(fn.MACD, macd),
            pl.Series(fn.MACD_SIGNAL, macd_signal),
            pl.Series(fn.MACD_HISTOGRAM, macd_histogram),
            pl.Series(fn.RSI, rsi),
        ]
    )
