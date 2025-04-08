import polars as pl

from fart.common.constants import DATETIME, TIMESTAMP


def parse_timestamp_to_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse the timestamp column to a datetime column in the given Polars DataFrame.

    Parameters
    ----------
    - df (pl.DataFrame): Polars DataFrame containing timestamp data.

    Returns
    -------
    - pl.DataFrame: Polars DataFrame with the timestamp column parsed to a datetime
    column.

    """

    return df.with_columns(
        [
            pl.from_epoch(df[TIMESTAMP], time_unit="ms").alias(DATETIME),
        ]
    )
