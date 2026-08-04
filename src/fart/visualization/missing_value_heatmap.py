from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.colors import ListedColormap

from fart.constants import IMPERIAL_RED_MAIN, PERSIAN_GREEN_MAIN, TIMESTAMP


def plot_missing_value_heatmap(df: pl.DataFrame, title: Optional[str] = None) -> None:
    """
    Plot a heatmap of missing values in `df`, one row per column, one
    column per record. If `df` has a `Timestamp` column, rows missing
    entirely from the data (gaps in the timestamp sequence) are also shown,
    not just missing values within existing rows.

    Parameters
    ----------
    - df (pl.DataFrame): DataFrame to inspect for missing values.
    - title (Optional[str]): Title to display above the heatmap.

    """
    df = fill_missing_candles(df)
    null_mask = df.select(pl.all().is_null()).to_numpy().T

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        figsize=(16, 4), constrained_layout=True
    )
    sns.heatmap(  # pyright: ignore[reportUnknownMemberType]
        null_mask,
        cmap=ListedColormap([PERSIAN_GREEN_MAIN, IMPERIAL_RED_MAIN]),
        cbar=False,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        square=False,
        rasterized=False,
        ax=ax,
    )
    if title:
        ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]

    plt.show()  # pyright: ignore[reportUnknownMemberType]


def fill_missing_candles(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reindex `df` on its inferred `Timestamp` cadence, inserting an explicit
    all-null row for every timestamp missing from the data. Sorts and
    deduplicates on `Timestamp` first, since real candle data can be out of
    order or contain duplicate timestamps (e.g. from a resumed download).
    A no-op if `df` has no `Timestamp` column or fewer than two rows.

    Parameters
    ----------
    - df (pl.DataFrame): Candle data with a `Timestamp` column.

    Returns
    -------
    - pl.DataFrame: `df` reindexed onto a complete, gap-free timestamp
    range, with an all-null row for every previously-missing timestamp.

    """
    if TIMESTAMP not in df.columns or len(df) < 2:
        return df

    df = df.sort(TIMESTAMP).unique(subset=[TIMESTAMP], keep="first").sort(TIMESTAMP)

    # The interval is taken as the most common gap (mode) rather than the
    # smallest one -- a single out-of-order or duplicate row would
    # otherwise corrupt detection via a negative or zero minimum.
    interval = df[TIMESTAMP].diff().drop_nulls().mode().min()
    if interval is None or interval <= 0:  # type: ignore
        return df

    start = df[TIMESTAMP].min()
    end = df[TIMESTAMP].max()
    full_timestamps = pl.DataFrame(
        {
            TIMESTAMP: pl.int_range(  # type: ignore
                start,
                end + interval,  # type: ignore
                step=interval,
                eager=True,
            )
        }
    )
    return full_timestamps.join(df, on=TIMESTAMP, how="left").sort(TIMESTAMP)
