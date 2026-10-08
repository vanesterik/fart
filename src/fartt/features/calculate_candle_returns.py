import polars as pl

from fartt.constants import CANDLE_RETURN, CLOSE


def calculate_candle_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate each candle's return: the signed percent change of Close from
    the previous candle, as a fraction. Sign carries direction (positive =
    price increase, negative = price decrease).

    Parameters
    ----------
    - df (pl.DataFrame): A DataFrame containing Close price data.

    Returns
    -------
    - pl.DataFrame: `df` with a `Return` column added.

    """
    return df.with_columns(pl.col(CLOSE).pct_change().alias(CANDLE_RETURN))
