import polars as pl

from fart.constants import CLOSE, MAGNITUDE


def calculate_magnitude(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the magnitude of Close price movements as the signed percent
    change between consecutive candles. Sign carries direction (positive =
    price increase, negative = price decrease).

    Parameters
    ----------
    - df (pl.DataFrame): A DataFrame containing Close price data.

    Returns
    -------
    - pl.DataFrame: `df` with a `Magnitude` column added.

    """
    return df.with_columns(pl.col(CLOSE).pct_change().alias(MAGNITUDE))
