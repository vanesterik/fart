from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fartt.constants import CANDLE_RETURN, DATETIME
from fartt.visualization.diverging_bar_chart import plot_diverging_bars


def plot_candle_returns(
    df: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    """
    Plot the candle return column (signed percent change of Close between
    consecutive candles) as a diverging bar chart over time -- bars rise
    above zero for up moves and fall below zero for down moves, colored by
    direction, to make the size and direction of every move easy to read
    at a glance.

    Parameters
    ----------
    - df (pl.DataFrame): Candle data with `Datetime` and `Return`
      columns (see `features/calculate_candle_returns.py`).
    - start (Optional[datetime]): Start of the period to plot. If omitted,
      plotting starts at the first record; if `end` is also omitted,
      plotting starts at the first record and runs to the last.
    - end (Optional[datetime]): End of the period to plot. If omitted,
      plotting runs through to the last record.

    """
    if start is not None and end is not None and end < start:
        raise ValueError(f"end ({end}) cannot be before start ({start}).")

    df_period = df
    if start is not None:
        df_period = df_period.filter(pl.col(DATETIME) >= start)
    if end is not None:
        df_period = df_period.filter(pl.col(DATETIME) <= end)

    df_valid = df_period.drop_nulls(CANDLE_RETURN)
    datetime_values = df_valid[DATETIME].to_numpy()
    candle_returns = df_valid[CANDLE_RETURN].to_numpy()

    width = 0.8
    if len(datetime_values) > 1:
        median_delta = np.median(np.diff(datetime_values))
        width = (median_delta / np.timedelta64(1, "D")) * 0.8

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(16, 4),
        constrained_layout=True,
    )
    plot_diverging_bars(ax, datetime_values, candle_returns, width=width)
    ax.set_title(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
        "Candle Returns (Signed Change of Close)"
    )
    ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
        CANDLE_RETURN
    )

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream
