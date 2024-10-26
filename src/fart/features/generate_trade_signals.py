# External imports
import polars as pl

# Internal imports
from fart.constants import classes as cl
from fart.constants import feature_names as fn
from fart.features.technical_indicators_config import TechnicalIndicatorsConfig


def generate_trade_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate hold, buy and sell signals by combining business rules of Bollinger
    Bands (BB), Exponential Moving Averages (EMA), Moving Average Convergence
    Divergence (MACD), and Relative Strength Index (RSI).

    ## Buy Signal Conditions:

    1. Price touches/breaks below lower BB, moves towards middle band
    2. 50-period EMA crosses above 200-period EMA ("golden cross")
    3. MACD line crosses above signal line (both below zero line)
    4. RSI moves from below oversold threshold (ie. 30) to above threshold plus
    ten (ie. 40)

    ## Sell Signal Conditions:

    1. Price touches/breaks above upper BB, moves towards middle band
    2. 50-period EMA crosses below 200-period EMA ("death cross")
    3. MACD line crosses below signal line (both above zero line)
    4. RSI moves from above overbought threshold (ie. 70) to below threshold
    minus ten (ie. 60)

    Params
    ------
    - df (pl.DataFrame): Input DataFrame

    Returns
    -------
    - df (pl.DataFrame): DataFrame with trade signals

    """

    # Initialize indicator configuration
    config = TechnicalIndicatorsConfig()

    return df.with_columns(
        [
            pl.when(
                # # Price touches/breaks below lower BB, moves towards middle band
                # (pl.col(fn.CLOSE) < pl.col(fn.BBANDS_LOWER))
                # & (pl.col(fn.CLOSE).shift(1) >= pl.col(fn.BBANDS_LOWER).shift(1))
                # # 50-period EMA crosses above 200-period EMA ("golden cross")
                # | (pl.col(fn.EMA_FAST) > pl.col(fn.EMA_SLOW))
                # & (pl.col(fn.EMA_FAST).shift(1) <= pl.col(fn.EMA_SLOW).shift(1))
                # # MACD line crosses above signal line (both below zero line)
                # | (pl.col(fn.MACD) > pl.col(fn.MACD_SIGNAL))
                # & (pl.col(fn.MACD).shift(1) <= pl.col(fn.MACD_SIGNAL).shift(1))
                # # RSI moves from below oversold threshold (ie. 30) to above threshold plus ten (ie. 40)
                # |
                (pl.col(fn.RSI) < config.rsi.oversold)
                & (pl.col(fn.RSI).shift(1) <= config.rsi.oversold)
            )
            .then(pl.lit(cl.BUY))
            .when(
                # # Price touches/breaks above upper BB, moves towards middle band
                # (pl.col(fn.CLOSE) > pl.col(fn.BBANDS_UPPER))
                # & (pl.col(fn.CLOSE).shift(1) <= pl.col(fn.BBANDS_UPPER).shift(1))
                # # 50-period EMA crosses below 200-period EMA ("death cross")
                # | (pl.col(fn.EMA_FAST) < pl.col(fn.EMA_SLOW))
                # & (pl.col(fn.EMA_FAST).shift(1) >= pl.col(fn.EMA_SLOW).shift(1))
                # # MACD line crosses below signal line (both above zero line)
                # | (pl.col(fn.MACD) < pl.col(fn.MACD_SIGNAL))
                # & (pl.col(fn.MACD).shift(1) >= pl.col(fn.MACD_SIGNAL).shift(1))
                # # RSI moves from above overbought threshold (ie. 70) to below threshold minus ten (ie. 60)
                # |
                (pl.col(fn.RSI) > config.rsi.overbought)
                & (pl.col(fn.RSI).shift(1) >= config.rsi.overbought)
            )
            .then(pl.lit(cl.SELL))
            .otherwise(pl.lit(cl.HOLD))
            .alias(fn.TRADE_SIGNAL),
        ]
    )
