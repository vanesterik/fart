import polars as pl

from fart.constants import TIMESTAMP


def sort_and_deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sort data by Timestamp and drop duplicate timestamps, keeping the first
    occurrence. Real candle data can be out of order or contain duplicate
    timestamps (e.g. from a resumed download), which would otherwise silently
    corrupt positional/rolling computations downstream (technical indicators,
    sliding return windows). A no-op if `df` has no `Timestamp` column.

    Parameters
    ----------
    - df (pl.DataFrame): Data with a `Timestamp` column.

    Returns
    -------
    - pl.DataFrame: `df` sorted by `Timestamp` with duplicate timestamps
    removed.

    """
    if TIMESTAMP not in df.columns:
        return df

    return df.sort(TIMESTAMP).unique(
        subset=[TIMESTAMP], keep="first", maintain_order=True
    )
