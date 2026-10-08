from collections.abc import Sequence
from dataclasses import dataclass

from fartt.exchange import Candle
from fartt.forecast.forecaster import Forecaster, forecast


@dataclass(frozen=True)
class Settings:
    """
    The analysis settings, from `fartt serve`'s options. Every value is a
    fraction: `0.0025` is 0.25%.

    """

    fee: float = 0.0025  # per leg: Bitvavo's entry tier, 0.25% taker
    slippage: float = 0.001  # per leg
    threshold: float | None = None  # None: the round-trip cost
    hit_rate_window: int = 100  # recent candles the hit rate covers

    @property
    def round_trip_cost(self) -> float:
        """Fee and slippage on both legs of a trade."""
        return 2 * (self.fee + self.slippage)


@dataclass(frozen=True)
class Analysis:
    """The latest forecast after costs, with the forecaster's hit rate."""

    model: str
    expected_return: float
    based_on_ms: int
    applies_to_ms: int
    round_trip_cost: float
    expected_return_after_costs: float
    threshold: float
    clears_entry_threshold: bool
    clears_exit_threshold: bool
    hit_rate: float | None  # None when no candle could be scored
    hit_rate_candles: int


def analyze(
    forecaster: Forecaster,
    candles: Sequence[Candle],
    interval_ms: int,
    settings: Settings,
) -> Analysis:
    """
    Analyse the forecast for the candle after the newest of `candles`
    (closed, oldest first). The threshold flags use `calculate_trade_returns`'
    rules, so the analysis and the backtest agree. Pass
    `forecaster.required_candles + settings.hit_rate_window` candles for a
    full hit-rate window; with fewer, the window shortens.

    """
    latest = forecast(forecaster, candles, interval_ms)
    cost = settings.round_trip_cost
    threshold = cost if settings.threshold is None else settings.threshold
    hit_rate, hit_rate_candles = _hit_rate(
        forecaster, candles, settings.hit_rate_window
    )
    return Analysis(
        model=latest.model,
        expected_return=latest.expected_return,
        based_on_ms=latest.based_on_ms,
        applies_to_ms=latest.applies_to_ms,
        round_trip_cost=cost,
        expected_return_after_costs=latest.expected_return - cost,
        threshold=threshold,
        clears_entry_threshold=latest.expected_return > threshold,
        clears_exit_threshold=latest.expected_return < -threshold,
        hit_rate=hit_rate,
        hit_rate_candles=hit_rate_candles,
    )


def _hit_rate(
    forecaster: Forecaster, candles: Sequence[Candle], window: int
) -> tuple[float | None, int]:
    """
    Score the forecaster's predicted direction on each of the last `window`
    candles, predicting each from the `required_candles` before it. Skipped:
    candles inside the training data (the rate stays out-of-sample), and
    those where the prediction or the realised return is exactly 0.

    """
    needed = forecaster.required_candles  # >= 1: forecast() checked it
    trained_through_ms = forecaster.trained_through_ms
    hits = scored = 0
    for i in range(max(needed, len(candles) - window), len(candles)):
        target = candles[i]
        if trained_through_ms is not None and target.timestamp <= trained_through_ms:
            continue
        realised = target.close / candles[i - 1].close - 1
        predicted = forecaster.predict(candles[i - needed : i])
        if realised == 0 or predicted == 0:
            continue
        scored += 1
        if (predicted > 0) == (realised > 0):
            hits += 1
    return (hits / scored if scored else None), scored
