from typing import List, Tuple

import polars as pl

# Internal imports
from fart.common.constants import (
    BBANDS_LOWER,
    BBANDS_LOWER_BOUNCE,
    BBANDS_UPPER,
    BBANDS_UPPER_BOUNCE,
    BUY_CLASS,
    CLOSE,
    EMA_DEATH_CROSS,
    EMA_FAST,
    EMA_GOLDEN_CROSS,
    EMA_SLOW,
    HOLD_CLASS,
    MACD,
    MACD_BEARISH_CROSS,
    MACD_BULLISH_CROSS,
    MACD_SIGNAL,
    RSI,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SELL_CLASS,
    TIMESTAMP,
    TRADE_SIGNAL,
)
from fart.features.technical_indicators_config import TechnicalIndicatorsConfig


class TradeSignals:
    def __init__(self, df: pl.DataFrame) -> None:
        """
        Initialize the TradeSignals class containing methods to generate trade
        signals based on technical indicators.

        Parameters
        ----------
        - df (pl.DataFrame): DataFrame containing the necessary columns.

        """

        self._df = df
        self._entry: Tuple[float, float] = (0, 0)
        self._exit: Tuple[float, float] = (0, 0)
        self._is_open_position = False
        self._positive_trades: List[float] = []

    @property
    def df(self) -> pl.DataFrame:
        """
        Returns the DataFrame containing the trade signals.

        """
        return self._df

    def generate(self) -> None:
        """
        Generate hold, buy and sell signals by combining business rules of
        Bollinger Bands (BB), Exponential Moving Averages (EMA), Moving Average
        Convergence Divergence (MACD), and Relative Strength Index (RSI).

        ## Buy Signal Conditions:

        1. Price touches/breaks below lower BB, moves towards middle band
        2. 50-period EMA crosses above 200-period EMA ("golden cross")
        3. MACD line crosses above signal line (both below zero line)
        4. RSI moves from below oversold threshold (ie. 30) to above threshold
           plus ten (ie. 40)

        ## Sell Signal Conditions:

        1. Price touches/breaks above upper BB, moves towards middle band
        2. 50-period EMA crosses below 200-period EMA ("death cross")
        3. MACD line crosses below signal line (both above zero line)
        4. RSI moves from above overbought threshold (ie. 70) to below threshold
           minus ten (ie. 60)

        """

        # Initialize indicator configuration
        config = TechnicalIndicatorsConfig()

        # Apply rules for generating trade signals
        self._df = self.df.with_columns(
            [
                # Bollinger Bands
                # Lower Bounce
                pl.when(
                    (pl.col(CLOSE) < pl.col(BBANDS_LOWER))
                    & (pl.col(CLOSE).shift(1) >= pl.col(BBANDS_LOWER).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(BBANDS_LOWER_BOUNCE),
                #
                # Upper Bounce
                pl.when(
                    (pl.col(CLOSE) > pl.col(BBANDS_UPPER))
                    & (pl.col(CLOSE).shift(1) <= pl.col(BBANDS_UPPER).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(BBANDS_UPPER_BOUNCE),
                #
                # Exponential Moving Average
                # Golden Cross
                pl.when(
                    (pl.col(EMA_FAST) > pl.col(EMA_SLOW))
                    & (pl.col(EMA_FAST).shift(1) <= pl.col(EMA_SLOW).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(EMA_GOLDEN_CROSS),
                #
                # Death Cross
                pl.when(
                    (pl.col(EMA_FAST) < pl.col(EMA_SLOW))
                    & (pl.col(EMA_FAST).shift(1) >= pl.col(EMA_SLOW).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(EMA_DEATH_CROSS),
                #
                # Moving Average Convergence Divergence
                # Bullish Cross
                pl.when(
                    (pl.col(MACD) > pl.col(MACD_SIGNAL))
                    & (pl.col(MACD).shift(1) <= pl.col(MACD_SIGNAL).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(MACD_BULLISH_CROSS),
                #
                # Bearish Cross
                pl.when(
                    (pl.col(MACD) < pl.col(MACD_SIGNAL))
                    & (pl.col(MACD).shift(1) >= pl.col(MACD_SIGNAL).shift(1))
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(MACD_BEARISH_CROSS),
                #
                # Relative Strength Index
                # Oversold
                pl.when(
                    (pl.col(RSI) <= config.rsi.oversold)
                    & (pl.col(RSI).shift(1) >= config.rsi.oversold + config.rsi.margin)
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(RSI_OVERSOLD),
                #
                # Overbought
                pl.when(
                    (pl.col(RSI) >= config.rsi.overbought)
                    & (
                        pl.col(RSI).shift(1)
                        <= config.rsi.overbought - config.rsi.margin
                    )
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(RSI_OVERBOUGHT),
            ]
        )

        # Apply technical indicators based on specific conditions
        self._df = self._df.with_columns(
            [
                pl.when(
                    (pl.col(EMA_GOLDEN_CROSS) == 1)
                    | (pl.col(BBANDS_LOWER_BOUNCE) == 1)
                    | (pl.col(MACD_BULLISH_CROSS) == 1)
                    | (pl.col(RSI_OVERSOLD) == 1)
                )
                .then(pl.lit(BUY_CLASS))
                .when(
                    (pl.col(EMA_DEATH_CROSS) == 1)
                    | (pl.col(BBANDS_UPPER_BOUNCE) == 1)
                    | (pl.col(MACD_BEARISH_CROSS) == 1)
                    | (pl.col(RSI_OVERBOUGHT) == 1)
                )
                .then(pl.lit(SELL_CLASS))
                .otherwise(pl.lit(HOLD_CLASS))
                .alias(TRADE_SIGNAL),
            ]
        )

    def optimize(self) -> None:
        """
        Optimize the trade signals by removing each buy/sell pair are yielding a
        negative result. This will make sure the trade signals are not
        generating a loss.

        """

        iteration_df = self._df
        iteration_count = 0

        while True:

            optimized_df = self._filter_positive_trades(iteration_df)
            iteration_count += 1

            # Check if the optimized DataFrame is equal to the iteration
            # DataFrame. If so, break the loop
            if optimized_df[TRADE_SIGNAL].equals(iteration_df[TRADE_SIGNAL]):
                break

            iteration_df = optimized_df

            # Raise exception as safety check to prevent infinite loops
            if iteration_count > 100:
                raise Exception("Optimization loop exceeded 100 iterations")

        # Update the original DataFrame with the optimized trade signals
        self._df = self._df.with_columns([optimized_df[TRADE_SIGNAL]])

    def _filter_positive_trades(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter positive trades from the DataFrame, by iterating over the
        DataFrame and determining if a buy/sell pair is yielding a positive
        result. In all other cases, the trade signals are set to hold which
        results in a clean trade signal series.

        Parameters
        ----------
        - df (pl.DataFrame): DataFrame containing close prices and trade signals

        Returns
        -------
        - pl.DataFrame: DataFrame with optimized trade signals

        """

        # Collect timestamps of negative trades
        for row in df.rows(named=True):
            if row[TRADE_SIGNAL] == BUY_CLASS and not self._is_open_position:
                self._is_open_position = True
                self._entry = (row[TIMESTAMP], float(row[CLOSE]))

            elif row[TRADE_SIGNAL] == SELL_CLASS and self._is_open_position:
                self._is_open_position = False
                self._exit = (row[TIMESTAMP], float(row[CLOSE]))

                # Destruct entry/exit timestamps/prices for easy access
                entry_timestamp, entry_price = self._entry
                exit_timestamp, exit_price = self._exit

                # Calculate profit/loss percentage
                profit_loss = (exit_price - entry_price) / entry_price

                # Collect the timestamps of the negative trades
                if profit_loss > 0:
                    self._positive_trades.extend([entry_timestamp, exit_timestamp])

        return df.with_columns(
            pl.when(pl.col(TIMESTAMP).is_in(pl.Series(self._positive_trades)))
            .then(pl.col(TRADE_SIGNAL))
            .otherwise(HOLD_CLASS)
            .alias(TRADE_SIGNAL)
        )
