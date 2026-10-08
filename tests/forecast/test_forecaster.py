from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from fartt.exchange import Candle
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


@dataclass(frozen=True)
class SpyForecaster:
    """Records the candles `predict` receives."""

    received: list[Sequence[Candle]] = field(default_factory=list)
    name: str = "spy"
    kind: str = "test double"
    required_candles: int = 3
    trained_through_ms: int | None = None

    def predict(self, candles: Sequence[Candle]) -> float:
        self.received.append(candles)
        return 0.0


def test_forecast_passes_predict_exactly_the_newest_required_candles() -> None:
    spy = SpyForecaster()
    candles = [candle(h) for h in range(200)]

    result = forecast(spy, candles, interval_ms=HOUR_MS)

    assert [c.timestamp for c in spy.received[0]] == [
        197 * HOUR_MS,
        198 * HOUR_MS,
        199 * HOUR_MS,
    ]
    assert result.based_on_ms == 199 * HOUR_MS


@pytest.mark.parametrize("required", [0, -1])
def test_forecast_rejects_a_forecaster_that_needs_no_candles(required: int) -> None:
    # A forecast is based on its newest candle, so it needs at least one.
    spy = SpyForecaster(required_candles=required)

    with pytest.raises(ValueError, match="at least 1"):
        forecast(spy, [candle(h) for h in range(5)], interval_ms=HOUR_MS)


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
