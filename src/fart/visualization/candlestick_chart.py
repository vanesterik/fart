from typing import Any

import mplfinance as mpf
import polars as pl

from fart.constants import (
    BBANDS,
    BBANDS_LOWER,
    BBANDS_MIDDLE,
    BBANDS_UPPER,
    DATETIME,
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

CONTOUR_LINE_ALPHA = 0.3


def plot_candlestick_chart(
    df: pl.DataFrame, timestamp: int | None = None, window_size: int = 120
) -> None:
    """
    Plot a candlestick chart with Bollinger Bands, Exponential Moving
    Averages, MACD, and RSI overlays. `df` must already have technical
    indicators computed (see `features/calculate_technical_indicators.py`)
    and a `Datetime` column (see `features/parse_timestamp_to_datetime.py`).

    Parameters
    ----------
    - df (pl.DataFrame): Candle data with technical indicators and a
      `Datetime` column.
    - timestamp (Optional[int]): Timestamp to center the plotted window on.
      Defaults to the most recent `window_size` candles.
    - window_size (int): Number of candles to plot.

    """
    data_window = _slice_data_window(df, timestamp, window_size)
    indicators = _build_indicator_plots(data_window)

    candles = data_window.to_pandas().set_index(  # pyright: ignore[reportUnknownMemberType] -- DataFrame.set_index's `keys` param is untyped upstream
        DATETIME
    )
    mpf.plot(
        candles,
        addplot=indicators,
        figsize=(24, 13.5),
        style="tradingview",
        type="hollow_and_filled",
    )


def _slice_data_window(
    df: pl.DataFrame, timestamp: int | None, window_size: int
) -> pl.DataFrame:
    # Filter DataFrame based half of window size before and after the
    # timestamp, but making sure to not go out of bounds at the start or end
    # of the DataFrame when passed timestamp is too close to the start or
    # end of the data.
    if timestamp is None:
        return df.tail(window_size)

    center_index = df[TIMESTAMP].search_sorted(timestamp)
    start = max(0, center_index - window_size // 2)
    end = start + window_size
    if end > len(df):
        end = len(df)
        start = max(0, end - window_size)
    return df[start:end]


def _build_indicator_plots(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        *_bbands_addplot(df),
        *_ema_addplot(df),
        *_macd_addplot(df),
        *_rsi_addplot(df),
    ]


def _bbands_addplot(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        mpf.make_addplot(
            df[BBANDS_MIDDLE].to_numpy(),
            color=HONOLULU_BLUE,
            fill_between=dict(
                alpha=0.1,
                color=HONOLULU_BLUE,
                y1=df[BBANDS_LOWER].to_numpy(),
                y2=df[BBANDS_UPPER].to_numpy(),
            ),
            label=BBANDS,
            panel=0,
        ),
        mpf.make_addplot(
            df[BBANDS_UPPER].to_numpy(),
            alpha=CONTOUR_LINE_ALPHA,
            color=HONOLULU_BLUE,
            panel=0,
        ),
        mpf.make_addplot(
            df[BBANDS_LOWER].to_numpy(),
            alpha=CONTOUR_LINE_ALPHA,
            color=HONOLULU_BLUE,
            panel=0,
        ),
    ]


def _ema_addplot(df: pl.DataFrame) -> list[dict[str, Any]]:
    ema_fast = df[EMA_FAST].to_numpy()
    ema_slow = df[EMA_SLOW].to_numpy()
    return [
        mpf.make_addplot(
            ema_fast,
            alpha=CONTOUR_LINE_ALPHA,
            color=HONOLULU_BLUE,
            fill_between=[
                dict(
                    alpha=CONTOUR_LINE_ALPHA,
                    color=PERSIAN_GREEN_MAIN,
                    y1=ema_fast,
                    y2=ema_slow,
                    where=ema_fast > ema_slow,
                ),
                dict(
                    alpha=CONTOUR_LINE_ALPHA,
                    color=IMPERIAL_RED_MAIN,
                    y1=ema_fast,
                    y2=ema_slow,
                    where=ema_fast < ema_slow,
                ),
            ],
            label=EMA_FAST,
            panel=0,
        ),
        mpf.make_addplot(
            ema_slow,
            alpha=CONTOUR_LINE_ALPHA,
            color=HONOLULU_BLUE,
            label=EMA_SLOW,
            panel=0,
        ),
    ]


def _macd_addplot(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        mpf.make_addplot(
            df[MACD].to_numpy(),
            color=YELLOW_SEA,
            panel=1,
        ),
        mpf.make_addplot(
            df[MACD_SIGNAL].to_numpy(),
            color=HONOLULU_BLUE,
            panel=1,
        ),
        mpf.make_addplot(
            df[MACD_HISTOGRAM].to_numpy(),
            color=_macd_histogram_colors(df),
            ylabel=MACD,
            panel=1,
            type="bar",
        ),
    ]


def _rsi_addplot(df: pl.DataFrame) -> list[dict[str, Any]]:
    config = TechnicalIndicatorsConfig()
    rsi_upper_bound = [config.rsi.overbought for _ in range(len(df))]
    rsi_lower_bound = [config.rsi.oversold for _ in range(len(df))]

    return [
        mpf.make_addplot(
            df[RSI].to_numpy(),
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
            alpha=CONTOUR_LINE_ALPHA,
            color=IMPERIAL_RED_MAIN,
            panel=2,
            secondary_y=False,
        ),
        mpf.make_addplot(
            rsi_lower_bound,
            alpha=CONTOUR_LINE_ALPHA,
            color=IMPERIAL_RED_MAIN,
            panel=2,
            secondary_y=False,
        ),
    ]


def _macd_histogram_colors(df: pl.DataFrame) -> list[str]:
    histogram = df[MACD_HISTOGRAM].to_numpy()
    colors: list[str] = []
    for i, current_value in enumerate(histogram):
        previous_value = histogram[i - 1] if i > 0 else 0

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
