import polars as pl

from fart.constants import CLOSE, MAGNITUDE
from fart.features.calculate_magnitude import calculate_magnitude


def test_calculate_magnitude_is_signed_percent_change() -> None:
    df = pl.DataFrame({CLOSE: [100.0, 110.0, 99.0]})

    result = calculate_magnitude(df)

    assert result[MAGNITUDE].to_list()[1:] == [0.1, -0.1]


def test_calculate_magnitude_first_row_is_null() -> None:
    df = pl.DataFrame({CLOSE: [100.0, 110.0]})

    result = calculate_magnitude(df)

    assert result[MAGNITUDE][0] is None
