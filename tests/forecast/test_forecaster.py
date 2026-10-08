import pytest

from fartt.forecast import Forecast, NotEnoughCandles, RepeatLastReturn, forecast
from tests.fakes import HOUR_MS, candle


def test_forecast_applies_to_the_candle_after_the_newest() -> None:
    candles = [candle(h, close=100.0 + h) for h in range(3)]

    result = forecast(RepeatLastReturn(), candles, interval_ms=HOUR_MS)

    assert result == Forecast(
        model="repeat-last-return",
        expected_return=pytest.approx(102.0 / 101.0 - 1),
        based_on_ms=2 * HOUR_MS,
        applies_to_ms=3 * HOUR_MS,
    )


def test_forecast_uses_only_the_newest_required_candles() -> None:
    # The first close would distort the result if it were used.
    candles = [candle(0, close=1.0)] + [candle(h, close=100.0) for h in range(1, 200)]

    result = forecast(RepeatLastReturn(), candles, interval_ms=HOUR_MS)

    assert result.expected_return == 0.0
    assert result.based_on_ms == 199 * HOUR_MS


def test_forecast_with_exactly_the_required_candles() -> None:
    candles = [candle(0, close=100.0), candle(1, close=100.0)]

    result = forecast(RepeatLastReturn(), candles, interval_ms=HOUR_MS)

    assert result.based_on_ms == HOUR_MS


def test_forecast_with_too_few_candles_raises_not_enough_candles() -> None:
    with pytest.raises(NotEnoughCandles) as raised:
        forecast(RepeatLastReturn(), [candle(0)], interval_ms=HOUR_MS)

    assert raised.value.needed == 2
    assert raised.value.got == 1
    assert "needs 2 candles" in str(raised.value)
    assert "got 1" in str(raised.value)


def test_forecast_without_candles_raises_not_enough_candles() -> None:
    with pytest.raises(NotEnoughCandles) as raised:
        forecast(RepeatLastReturn(), [], interval_ms=HOUR_MS)

    assert raised.value.got == 0
