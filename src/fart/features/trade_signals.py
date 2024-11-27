import time
from ast import expr
from os import close, times
from typing import Any, Dict, List, Tuple

import polars as pl

# Internal imports
from fart.constants import classes as cl
from fart.constants import feature_names as fn
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

        # Apply technical indicators based on specific conditions
        self._df = self._df.with_columns(
            [
                pl.when(
                    (
                        (pl.col(fn.EMA_FAST) > pl.col(fn.EMA_SLOW))
                        & (pl.col(fn.EMA_FAST).shift(1) <= pl.col(fn.EMA_SLOW).shift(1))
                    )
                    | (
                        (pl.col(fn.CLOSE) < pl.col(fn.BBANDS_LOWER))
                        & (
                            pl.col(fn.CLOSE).shift(1)
                            >= pl.col(fn.BBANDS_LOWER).shift(1)
                        )
                    )
                    | (
                        (pl.col(fn.MACD) > pl.col(fn.MACD_SIGNAL))
                        & (pl.col(fn.MACD).shift(1) <= pl.col(fn.MACD_SIGNAL).shift(1))
                    )
                    | (
                        (pl.col(fn.RSI) <= config.rsi.oversold)
                        & (pl.col(fn.RSI).shift(1) >= config.rsi.oversold + 10)
                    )
                )
                .then(pl.lit(cl.BUY))
                .when(
                    (
                        (pl.col(fn.EMA_FAST) < pl.col(fn.EMA_SLOW))
                        & (pl.col(fn.EMA_FAST).shift(1) >= pl.col(fn.EMA_SLOW).shift(1))
                    )
                    | (
                        (pl.col(fn.CLOSE) > pl.col(fn.BBANDS_UPPER))
                        & (
                            pl.col(fn.CLOSE).shift(1)
                            <= pl.col(fn.BBANDS_UPPER).shift(1)
                        )
                    )
                    | (
                        (pl.col(fn.MACD) < pl.col(fn.MACD_SIGNAL))
                        & (pl.col(fn.MACD).shift(1) >= pl.col(fn.MACD_SIGNAL).shift(1))
                    )
                    | (
                        (pl.col(fn.RSI) >= config.rsi.overbought)
                        & (pl.col(fn.RSI).shift(1) <= config.rsi.overbought - 10)
                    )
                )
                .then(pl.lit(cl.SELL))
                .otherwise(pl.lit(cl.HOLD))
                .alias(fn.TRADE_SIGNAL),
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
            if optimized_df[fn.TRADE_SIGNAL].equals(iteration_df[fn.TRADE_SIGNAL]):
                break

            iteration_df = optimized_df

            # Raise exception as safety check to prevent infinite loops
            if iteration_count > 100:
                raise Exception("Optimization loop exceeded 100 iterations")

        # Update the original DataFrame with the optimized trade signals
        self._df = self._df.with_columns([optimized_df[fn.TRADE_SIGNAL]])

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
            if row[fn.TRADE_SIGNAL] == cl.BUY and not self._is_open_position:
                self._is_open_position = True
                self._entry = (row[fn.TIMESTAMP], float(row[fn.CLOSE]))

            elif row[fn.TRADE_SIGNAL] == cl.SELL and self._is_open_position:
                self._is_open_position = False
                self._exit = (row[fn.TIMESTAMP], float(row[fn.CLOSE]))

                # Destruct entry/exit timestamps/prices for easy access
                entry_timestamp, entry_price = self._entry
                exit_timestamp, exit_price = self._exit

                # Calculate profit/loss percentage
                profit_loss = (exit_price - entry_price) / entry_price

                # Collect the timestamps of the negative trades
                if profit_loss > 0:
                    self._positive_trades.extend([entry_timestamp, exit_timestamp])

        return df.with_columns(
            pl.when(pl.col(fn.TIMESTAMP).is_in(pl.Series(self._positive_trades)))
            .then(pl.col(fn.TRADE_SIGNAL))
            .otherwise(cl.HOLD)
            .alias(fn.TRADE_SIGNAL)
        )
