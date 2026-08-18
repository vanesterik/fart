import polars as pl

from fart.constants import CLOSE, TIMESTAMP
from fart.features.sort_and_deduplicate import sort_and_deduplicate


def test_sort_and_deduplicate_sorts_out_of_order_rows() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [120_000, 0, 60_000],
            CLOSE: [3.0, 1.0, 2.0],
        }
    )

    sorted_df = sort_and_deduplicate(df)

    assert sorted_df[TIMESTAMP].to_list() == [0, 60_000, 120_000]
    assert sorted_df[CLOSE].to_list() == [1.0, 2.0, 3.0]


def test_sort_and_deduplicate_drops_duplicate_timestamps() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 60_000, 60_000, 120_000],
            CLOSE: [1.0, 2.0, 2.0, 3.0],
        }
    )

    deduped = sort_and_deduplicate(df)

    assert deduped[TIMESTAMP].to_list() == [0, 60_000, 120_000]
    assert deduped[CLOSE].to_list() == [1.0, 2.0, 3.0]


def test_sort_and_deduplicate_keeps_first_occurrence_on_duplicate() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 60_000, 60_000],
            CLOSE: [1.0, 2.0, 999.0],
        }
    )

    deduped = sort_and_deduplicate(df)

    assert deduped[CLOSE].to_list() == [1.0, 2.0]


def test_sort_and_deduplicate_does_not_fill_gaps() -> None:
    df = pl.DataFrame(
        {
            TIMESTAMP: [0, 180_000],
            CLOSE: [1.0, 4.0],
        }
    )

    result = sort_and_deduplicate(df)

    assert result[TIMESTAMP].to_list() == [0, 180_000]
    assert result[CLOSE].to_list() == [1.0, 4.0]


def test_sort_and_deduplicate_noop_without_timestamp_column() -> None:
    df = pl.DataFrame({CLOSE: [1.0, 2.0, 3.0]})

    assert sort_and_deduplicate(df).equals(df)
