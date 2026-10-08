import pytest

from fartt.forecast import RepeatLastReturn
from tests.fakes import candle


def test_repeat_last_return_predicts_the_last_candles_return() -> None:
    baseline = RepeatLastReturn()

    predicted = baseline.predict([candle(0, close=100.0), candle(1, close=101.0)])

    assert predicted == pytest.approx(0.01)


def test_repeat_last_return_is_negative_after_a_fall() -> None:
    baseline = RepeatLastReturn()

    predicted = baseline.predict([candle(0, close=100.0), candle(1, close=98.0)])

    assert predicted == pytest.approx(-0.02)


def test_repeat_last_return_uses_only_the_last_two_candles() -> None:
    baseline = RepeatLastReturn()
    candles = [candle(0, close=50.0), candle(1, close=100.0), candle(2, close=110.0)]

    assert baseline.predict(candles) == pytest.approx(0.1)


def test_repeat_last_return_describes_itself() -> None:
    baseline = RepeatLastReturn()

    assert baseline.name == "repeat-last-return"
    assert baseline.kind == "naive baseline"
    assert baseline.required_candles == 2
    assert baseline.trained_through_ms is None
