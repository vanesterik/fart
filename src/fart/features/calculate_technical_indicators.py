# External imports
import polars as pl
from talib import BBANDS as calculate_bbands
from talib import EMA as calculate_ema
from talib import MACD as calculate_macd
from talib import RSI as calculate_rsi

# Internal imports
from fart.constants import (
    BBANDS_LOWER,
    BBANDS_MIDDLE,
    BBANDS_UPPER,
    CLOSE,
    EMA_FAST,
    EMA_SLOW,
    MACD,
    MACD_HISTOGRAM,
    MACD_SIGNAL,
    RSI,
)
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
    close_prices = df[CLOSE].to_numpy()

    # Bollinger Bands
    bbands_upper, bbands_middle, bbands_lower = calculate_bbands(
        close_prices,
        timeperiod=config.bbands.period,
        nbdevdn=config.bbands.standard_deviation,
        nbdevup=config.bbands.standard_deviation,
    )

    # Exponential Moving Averages
    ema_fast = calculate_ema(close_prices, timeperiod=config.ema.fast_period)
    ema_slow = calculate_ema(close_prices, timeperiod=config.ema.slow_period)

    # Moving Average Convergence Divergence
    macd, macd_signal, macd_histogram = calculate_macd(
        close_prices,
        fastperiod=config.macd.fast_period,
        slowperiod=config.macd.slow_period,
        signalperiod=config.macd.signal_period,
    )

    # Relative Strength Index
    rsi = calculate_rsi(close_prices, timeperiod=config.rsi.period)

    # Return DataFrame with calculated indicators
    return df.with_columns(
        [
            pl.Series(BBANDS_UPPER, bbands_upper),
            pl.Series(BBANDS_MIDDLE, bbands_middle),
            pl.Series(BBANDS_LOWER, bbands_lower),
            pl.Series(EMA_FAST, ema_fast),
            pl.Series(EMA_SLOW, ema_slow),
            pl.Series(MACD, macd),
            pl.Series(MACD_SIGNAL, macd_signal),
            pl.Series(MACD_HISTOGRAM, macd_histogram),
            pl.Series(RSI, rsi),
        ]
    )
