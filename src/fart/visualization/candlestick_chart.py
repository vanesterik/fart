from typing import Dict, List, Optional

import mplfinance as mpf
import numpy as np
import pandas as pd

from fart.constants import colors as co
from fart.constants import feature_names as fn
from fart.features.technical_indicators_config import TechnicalIndicatorsConfig


class CandlestickChart:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the CandlestickChart class containing methods to plot a
        candlestick chart with various technical indicators.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        """

        # Assign passed DataFrame
        self._df = df

        # Initialize indicator configuration
        self._config = TechnicalIndicatorsConfig()

        # Define various plot configurations
        self._contour_line_alpha = 0.3
        self._marker = "o"
        self._marker_size = 100

    def plot(self, timestamp: Optional[int] = None) -> None:
        """
        Plot a candlestick chart based on the passed Pandas DataFrame. The plot
        will include various technical indicators such as Bollinger Bands,
        Exponential Moving Averages, Moving Average Convergence Divergence, and
        Relative Strength Index. These indicators are required and should be
        present in the passed data.

        Parameters
        ----------
        - timestamp (Optional[int]): Timestamp to plot the chart around (if
          available).

        """

        # Slice the data window based on the passed timestamp
        data_window = self._slice_data_window(timestamp)
        # Define indicator plots based on all technical indicators
        indicators = self._add_indicator_plots(data_window)
        # Add trade signals where applicable to the indicator plots
        indicators = self._add_trade_signals(data_window, indicators)

        # Plot the candlestick chart with all bells and whistles
        mpf.plot(
            data_window,
            addplot=indicators,
            figsize=(21, 15),
            style="tradingview",
            type="hollow_and_filled",
        )

    def _slice_data_window(self, timestamp: Optional[int] = None) -> pd.DataFrame:
        # Define maximum number of candles to plot, this because the plot can
        # get too crowded with too many candles on the chart at once and become
        # unreadable.
        window_size = 120

        # Filter DataFrame based 60 before and 60 after the timestamp, but
        # making sure to not go out of bounds at the start or end of the
        # DataFrame when passed timestamp is too close to the start or end of
        # the data.
        if timestamp:
            # Find the index of the timestamp in the DataFrame
            center_index = self._df[fn.TIMESTAMP].searchsorted(timestamp)
            # Define the start and end index of the window
            start = max(0, center_index - window_size // 2)
            end = start + window_size
            # Prevent going out of bounds
            if end > len(self._df):
                end = len(self._df)
                start = max(0, end - window_size)
            # Return the sliced DataFrame
            return self._df[start:end]

        # If no timestamp is provided, plot the last 120 candles
        else:
            return self._df.tail(window_size)

    def _add_indicator_plots(self, df: pd.DataFrame) -> List[mpf.make_addplot]:
        """
        Add various technical indicator plots to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - indicators (List[mpf.make_addplot]): List of technical indicator plots.

        """
        return [
            *self._add_bbands_plot(df),
            *self._add_ema_plot(df),
            *self._add_macd_plot(df),
            *self._add_rsi_plot(df),
        ]

    def _add_bbands_plot(self, df: pd.DataFrame) -> List[mpf.make_addplot]:
        """
        Add Bollinger Bands plot to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - indicators (List[mpf.make_addplot]): List of Bollinger Bands plot.

        """
        return [
            mpf.make_addplot(
                df[fn.BBANDS_MIDDLE],
                color=co.HONOLULU_BLUE,
                fill_between=dict(
                    alpha=0.1,
                    color=co.HONOLULU_BLUE,
                    y1=df[fn.BBANDS_LOWER].values,
                    y2=df[fn.BBANDS_UPPER].values,
                ),
                label=fn.BBANDS,
                panel=0,
            ),
            mpf.make_addplot(
                df[fn.BBANDS_UPPER],
                alpha=self._contour_line_alpha,
                color=co.HONOLULU_BLUE,
                panel=0,
            ),
            mpf.make_addplot(
                df[fn.BBANDS_LOWER],
                alpha=self._contour_line_alpha,
                color=co.HONOLULU_BLUE,
                panel=0,
            ),
        ]

    def _add_ema_plot(self, df: pd.DataFrame) -> List[mpf.make_addplot]:
        """
        Add Exponential Moving Averages plot to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - indicators (List[mpf.make_addplot]): List of Exponential Moving
          Averages plot.

        """
        return [
            mpf.make_addplot(
                df[fn.EMA_FAST],
                alpha=self._contour_line_alpha,
                color=co.HONOLULU_BLUE,
                fill_between=[
                    dict(
                        alpha=self._contour_line_alpha,
                        color=co.PERSIAN_GREEN_MAIN,
                        y1=df[fn.EMA_FAST].values,
                        y2=df[fn.EMA_SLOW].values,
                        where=df[fn.EMA_FAST] > df[fn.EMA_SLOW],
                    ),
                    dict(
                        alpha=self._contour_line_alpha,
                        color=co.IMPERIAL_RED_MAIN,
                        y1=df[fn.EMA_FAST].values,
                        y2=df[fn.EMA_SLOW].values,
                        where=df[fn.EMA_FAST] < df[fn.EMA_SLOW],
                    ),
                ],
                label=fn.EMA_FAST,
                panel=0,
            ),
            mpf.make_addplot(
                df[fn.EMA_SLOW],
                alpha=self._contour_line_alpha,
                color=co.HONOLULU_BLUE,
                label=fn.EMA_SLOW,
                panel=0,
            ),
        ]

    def _add_macd_plot(self, df: pd.DataFrame) -> List[mpf.make_addplot]:
        """
        Add Moving Average Convergence Divergence plot to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - indicators (List[mpf.make_addplot]): List of Moving Average
          Convergence Divergence plot.

        """
        return [
            mpf.make_addplot(
                df[fn.MACD],
                color=co.YELLOW_SEA,
                panel=1,
            ),
            mpf.make_addplot(
                df[fn.MACD_SIGNAL],
                color=co.HONOLULU_BLUE,
                panel=1,
            ),
            mpf.make_addplot(
                df[fn.MACD_HISTOGRAM],
                color=self._generate_macd_colors(df),
                ylabel=fn.MACD,
                panel=1,
                type="bar",
            ),
        ]

    def _add_rsi_plot(self, df: pd.DataFrame) -> List[mpf.make_addplot]:
        """
        Add Relative Strength Index plot to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - indicators (List[mpf.make_addplot]): List of Relative Strength Index
          plot.

        """

        # Generate RSI upper and lower bounds for the fill_between plot
        rsi_upper_bound = [self._config.rsi.overbought for _ in range(len(df))]
        rsi_lower_bound = [self._config.rsi.oversold for _ in range(len(df))]

        return [
            mpf.make_addplot(
                df[fn.RSI],
                color=co.IMPERIAL_RED_MAIN,
                fill_between=dict(
                    alpha=0.1,
                    color=co.IMPERIAL_RED_MAIN,
                    y1=rsi_upper_bound,
                    y2=rsi_lower_bound,
                ),
                ylabel=fn.RSI,
                panel=2,
            ),
            mpf.make_addplot(
                rsi_upper_bound,
                alpha=self._contour_line_alpha,
                color=co.IMPERIAL_RED_MAIN,
                panel=2,
                secondary_y=False,
            ),
            mpf.make_addplot(
                rsi_lower_bound,
                alpha=self._contour_line_alpha,
                color=co.IMPERIAL_RED_MAIN,
                panel=2,
                secondary_y=False,
            ),
        ]

    def _add_trade_signals(
        self, df: pd.DataFrame, indicators: List[mpf.make_addplot]
    ) -> List[mpf.make_addplot]:
        """
        Add trade signals to the candlestick chart.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.
        - indicator (List[mpf.make_addplot]): List of trade signals to plot.

        Returns
        -------
        - indicator (List[mpf.make_addplot]): List of trade signals with
          conditionally added buy and sell signals.

        """

        buy_signals = [
            (df[fn.CLOSE].iloc[i] if df[fn.TRADE_SIGNAL].iloc[i] == 1 else np.NaN)
            for i in range(len(df))
        ]
        sell_signals = [
            (df[fn.CLOSE].iloc[i] if df[fn.TRADE_SIGNAL].iloc[i] == -1 else np.NaN)
            for i in range(len(df))
        ]

        if buy_signals.count(np.NaN) != len(buy_signals):
            indicators.append(
                mpf.make_addplot(
                    buy_signals,
                    color=co.GREEN,
                    edgecolors=co.WHITE,
                    marker=self._marker,
                    markersize=self._marker_size,
                    panel=0,
                    scatter=True,
                )
            )

        if sell_signals.count(np.NaN) != len(sell_signals):
            indicators.append(
                mpf.make_addplot(
                    sell_signals,
                    color=co.RED,
                    edgecolors=co.WHITE,
                    marker=self._marker,
                    markersize=self._marker_size,
                    panel=0,
                    scatter=True,
                )
            )

        return indicators

    def _generate_macd_colors(self, df: pd.DataFrame) -> List[str]:
        """
        Generate colors for the MACD histogram plot.

        Parameters
        ----------
        - df (pd.DataFrame): Pandas DataFrame containing candle data.

        Returns
        -------
        - colors (List[str]): List of colors for the MACD histogram plot.

        """

        colors = []
        for i in range(len(df)):
            current_value = df[fn.MACD_HISTOGRAM].iloc[i]
            previous_value = df[fn.MACD_HISTOGRAM].iloc[i - 1] if i > 0 else 0

            if current_value >= 0:
                colors.append(
                    co.PERSIAN_GREEN_MAIN
                    if previous_value < current_value
                    else co.PERSIAN_GREEN_LIGHT
                )
            else:
                colors.append(
                    co.IMPERIAL_RED_MAIN
                    if previous_value > current_value
                    else co.IMPERIAL_RED_LIGHT
                )

        return colors