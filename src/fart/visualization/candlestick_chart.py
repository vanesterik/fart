from typing import List, Optional

import mplfinance as mpf
import pandas as pd

from fart.constants import (
    BBANDS,
    BBANDS_LOWER,
    BBANDS_MIDDLE,
    BBANDS_UPPER,
    EMA_FAST,
    EMA_SLOW,
    HONOLULU_BLUE,
    IMPERIAL_RED_LIGHT,
    IMPERIAL_RED_MAIN,
    MACD,
    MACD_HISTOGRAM,
    MACD_SIGNAL,
    PERSIAN_GREEN_LIGHT,
    PERSIAN_GREEN_MAIN,
    RSI,
    TIMESTAMP,
    YELLOW_SEA,
)
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

    def plot(self, timestamp: Optional[int] = None, window_size: int = 120) -> None:
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
        - window_size (int): Number of candles to plot. This because the plot
          can get too crowded with too many candles on the chart at once and
          become unreadable.

        """

        # Slice the data window based on the passed timestamp
        data_window = self._slice_data_window(timestamp, window_size)
        # Define indicator plots based on all technical indicators
        indicators = self._add_indicator_plots(data_window)

        # Plot the candlestick chart with all bells and whistles
        mpf.plot(
            data_window,
            addplot=indicators,
            figsize=(24, 13.5),
            style="tradingview",
            type="hollow_and_filled",
        )

    def _slice_data_window(
        self,
        timestamp: Optional[int] = None,
        window_size: int = 120,
    ) -> pd.DataFrame:
        # Filter DataFrame based half of window size before and after the
        # timestamp, but making sure to not go out of bounds at the start or end
        # of the DataFrame when passed timestamp is too close to the start or
        # end of the data.
        if timestamp:
            # Find the index of the timestamp in the DataFrame
            center_index = self._df[TIMESTAMP].searchsorted(timestamp)
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
                df[BBANDS_MIDDLE],
                color=HONOLULU_BLUE,
                fill_between=dict(
                    alpha=0.1,
                    color=HONOLULU_BLUE,
                    y1=df[BBANDS_LOWER].values,
                    y2=df[BBANDS_UPPER].values,
                ),
                label=BBANDS,
                panel=0,
            ),
            mpf.make_addplot(
                df[BBANDS_UPPER],
                alpha=self._contour_line_alpha,
                color=HONOLULU_BLUE,
                panel=0,
            ),
            mpf.make_addplot(
                df[BBANDS_LOWER],
                alpha=self._contour_line_alpha,
                color=HONOLULU_BLUE,
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
                df[EMA_FAST],
                alpha=self._contour_line_alpha,
                color=HONOLULU_BLUE,
                fill_between=[
                    dict(
                        alpha=self._contour_line_alpha,
                        color=PERSIAN_GREEN_MAIN,
                        y1=df[EMA_FAST].values,
                        y2=df[EMA_SLOW].values,
                        where=df[EMA_FAST] > df[EMA_SLOW],
                    ),
                    dict(
                        alpha=self._contour_line_alpha,
                        color=IMPERIAL_RED_MAIN,
                        y1=df[EMA_FAST].values,
                        y2=df[EMA_SLOW].values,
                        where=df[EMA_FAST] < df[EMA_SLOW],
                    ),
                ],
                label=EMA_FAST,
                panel=0,
            ),
            mpf.make_addplot(
                df[EMA_SLOW],
                alpha=self._contour_line_alpha,
                color=HONOLULU_BLUE,
                label=EMA_SLOW,
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
                df[MACD],
                color=YELLOW_SEA,
                panel=1,
            ),
            mpf.make_addplot(
                df[MACD_SIGNAL],
                color=HONOLULU_BLUE,
                panel=1,
            ),
            mpf.make_addplot(
                df[MACD_HISTOGRAM],
                color=self._create_macd_colors(df),
                ylabel=MACD,
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
                df[RSI],
                color=IMPERIAL_RED_MAIN,
                fill_between=dict(
                    alpha=0.1,
                    color=IMPERIAL_RED_MAIN,
                    y1=rsi_upper_bound,
                    y2=rsi_lower_bound,
                ),
                ylabel=RSI,
                panel=2,
            ),
            mpf.make_addplot(
                rsi_upper_bound,
                alpha=self._contour_line_alpha,
                color=IMPERIAL_RED_MAIN,
                panel=2,
                secondary_y=False,
            ),
            mpf.make_addplot(
                rsi_lower_bound,
                alpha=self._contour_line_alpha,
                color=IMPERIAL_RED_MAIN,
                panel=2,
                secondary_y=False,
            ),
        ]

    def _create_macd_colors(self, df: pd.DataFrame) -> List[str]:
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
            current_value = df[MACD_HISTOGRAM].iloc[i]
            previous_value = df[MACD_HISTOGRAM].iloc[i - 1] if i > 0 else 0

            if current_value >= 0:
                colors.append(
                    PERSIAN_GREEN_MAIN
                    if previous_value < current_value
                    else PERSIAN_GREEN_LIGHT
                )
            else:
                colors.append(
                    IMPERIAL_RED_MAIN
                    if previous_value > current_value
                    else IMPERIAL_RED_LIGHT
                )

        return colors
