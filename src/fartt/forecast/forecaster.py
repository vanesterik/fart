from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from fartt.exchange import Candle


class Forecaster(Protocol):
    """
    Structural interface for a model that forecasts the next candle's
    return -- no inheritance required. `RepeatLastReturn` (`baseline.py`) is
    the naive baseline; epic B's trained model is a second implementation.

    The model only predicts: timestamps, freshness and costs are handled
    around it (`forecast()`, `analysis.py`), so a model has one method to
    implement. The descriptive members are read-only properties, so frozen
    dataclasses and plain classes both satisfy the Protocol.

    """

    @property
    def name(self) -> str:
        """Model name, e.g. `"repeat-last-return"`."""
        ...

    @property
    def kind(self) -> str:
        """What sort of model, e.g. `"naive baseline"` or `"trained model"`."""
        ...

    @property
    def required_candles(self) -> int:
        """How many closed candles `predict` needs."""
        ...

    @property
    def trained_through_ms(self) -> int | None:
        """
        Open time of the last candle in the training data, or `None` for a
        model that isn't trained. The hit rate only scores later candles,
        so it stays out-of-sample.

        """
        ...

    def predict(self, candles: Sequence[Candle]) -> float:
        """
        Return the expected return of the candle after the last one given,
        as a fraction (`0.004` = +0.4%). `candles` are closed, oldest first,
        and exactly `required_candles` long.

        """
        ...


@dataclass(frozen=True)
class Forecast:
    """A forecaster's expected return for one candle, with its timing."""

    model: str
    expected_return: float  # fraction: 0.004 = +0.4%
    based_on_ms: int  # open time of the newest candle the forecast used
    applies_to_ms: int  # open time of the candle being forecast


class NotEnoughCandles(Exception):
    """The forecaster needs more closed candles than were available."""

    def __init__(self, needed: int, got: int) -> None:
        super().__init__(f"The forecaster needs {needed} candles but got {got}")
        self.needed = needed
        self.got = got


def forecast(
    forecaster: Forecaster, candles: Sequence[Candle], interval_ms: int
) -> Forecast:
    """
    Forecast the candle after the newest of `candles` (closed, oldest
    first), using only the last `forecaster.required_candles` of them.

    """
    needed = forecaster.required_candles
    if len(candles) < needed:
        raise NotEnoughCandles(needed, len(candles))

    window = candles[-needed:]
    based_on_ms = window[-1].timestamp
    return Forecast(
        model=forecaster.name,
        expected_return=forecaster.predict(window),
        based_on_ms=based_on_ms,
        applies_to_ms=based_on_ms + interval_ms,
    )
