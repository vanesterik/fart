import polars as pl

from fart.constants import CLOSE, TIMESTAMP
from fart.visualization.missing_value_heatmap import fill_missing_candles


def test_fill_missing_candles_inserts_null_row_for_gap() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 60_000, 180_000],
            CLOSE: [1.0, 2.0, 4.0],
        }
    )

    filled = fill_missing_candles(df)

    assert filled[TIMESTAMP].to_list() == [0, 60_000, 120_000, 180_000]
    assert filled[CLOSE].to_list() == [1.0, 2.0, None, 4.0]


def test_fill_missing_candles_sorts_out_of_order_rows() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [120_000, 0, 60_000],
            CLOSE: [3.0, 1.0, 2.0],
        }
    )

    filled = fill_missing_candles(df)

    assert filled[TIMESTAMP].to_list() == [0, 60_000, 120_000]
    assert filled[CLOSE].to_list() == [1.0, 2.0, 3.0]


def test_fill_missing_candles_drops_duplicate_timestamps() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 60_000, 60_000, 120_000],
            CLOSE: [1.0, 2.0, 2.0, 3.0],
        }
    )

    filled = fill_missing_candles(df)

    assert filled[TIMESTAMP].to_list() == [0, 60_000, 120_000]
    assert filled[CLOSE].to_list() == [1.0, 2.0, 3.0]


def test_fill_missing_candles_ignores_outlier_diff_when_inferring_cadence() -> None:
    # A rare, irregular gap could be mistaken for the true cadence if picked
    # via the minimum diff; the dominant, most common gap (mode) should be
    # used instead so one irregular row doesn't corrupt gap detection for
    # the rest of the series.
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 60_000, 120_000, 180_000, 180_500, 300_000],
            CLOSE: [1.0, 2.0, 3.0, 4.0, 4.5, 6.0],
        }
    )

    filled = fill_missing_candles(df)

    assert filled[TIMESTAMP].to_list() == [
        0,
        60_000,
        120_000,
        180_000,
        240_000,
        300_000,
    ]
    assert filled[CLOSE].to_list() == [1.0, 2.0, 3.0, 4.0, None, 6.0]


def test_fill_missing_candles_noop_without_timestamp_column() -> None:
    df = pl.DataFrame({CLOSE: [1.0, 2.0, 3.0]})

    assert fill_missing_candles(df).equals(df)


def test_fill_missing_candles_noop_with_fewer_than_two_rows() -> None:
    df = pl.DataFrame({TIMESTAMP: [0], CLOSE: [1.0]})

    assert fill_missing_candles(df).equals(df)
