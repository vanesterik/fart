import polars as pl

from fartt.constants import CANDLE_RETURN, CLOSE
from fartt.features.calculate_candle_returns import calculate_candle_returns


def test_calculate_candle_returns_is_signed_percent_change() -> None:
    df = pl.DataFrame({CLOSE: [100.0, 110.0, 99.0]})

    result = calculate_candle_returns(df)

    assert result[CANDLE_RETURN].to_list()[1:] == [0.1, -0.1]


def test_calculate_candle_returns_first_row_is_null() -> None:
    df = pl.DataFrame({CLOSE: [100.0, 110.0]})

    result = calculate_candle_returns(df)

    assert result[CANDLE_RETURN][0] is None
