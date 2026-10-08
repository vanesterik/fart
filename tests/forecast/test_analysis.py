from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from fartt.exchange import Candle
from fartt.features.calculate_trade_returns import calculate_trade_returns
from fartt.forecast import (
    Analysis,
    NotEnoughCandles,
    RepeatLastReturn,
    Settings,
    analyze,
)
from tests.fakes import HOUR_MS, candle


@dataclass(frozen=True)
class ConstantForecaster:
    value: float
    name: str = "constant"
    kind: str = "test"
    required_candles: int = 2
    trained_through_ms: int | None = None

    def predict(self, candles: Sequence[Candle]) -> float:
        return self.value


@dataclass
class RecordingForecaster:
    """Records which candles (by hour) each `predict` call was given."""

    windows: list[tuple[int, ...]] = field(default_factory=list[tuple[int, ...]])
    name: str = "recording"
    kind: str = "test"
    required_candles: int = 2
    trained_through_ms: int | None = None

    def predict(self, candles: Sequence[Candle]) -> float:
        self.windows.append(tuple(c.timestamp // HOUR_MS for c in candles))
        return 0.01


def _closes(*closes: float) -> list[Candle]:
    return [candle(hour, close=close) for hour, close in enumerate(closes)]


# Returns: +1%, +0.99%, -0.98%, -0.99%, 0, +1%. Repeating the last return,
# candle 2 is a hit, 3 a miss, 4 a hit; 5 (flat) and 6 (predicted 0) are
# skipped.
MIXED = _closes(100, 101, 102, 101, 100, 100, 101)


def test_analyze_computes_the_after_cost_numbers() -> None:
    analysis = analyze(RepeatLastReturn(), _closes(100, 101), HOUR_MS, Settings())

    assert analysis.model == "repeat-last-return"
    assert analysis.expected_return == pytest.approx(0.01)
    assert analysis.round_trip_cost == pytest.approx(0.007)
    assert analysis.expected_return_after_costs == pytest.approx(0.003)
    assert analysis.threshold == pytest.approx(0.007)  # the round-trip cost
    assert analysis.clears_entry_threshold is True
    assert analysis.clears_exit_threshold is False
    assert analysis.based_on_ms == 1 * HOUR_MS
    assert analysis.applies_to_ms == 2 * HOUR_MS


def test_analyze_uses_an_explicit_threshold() -> None:
    settings = Settings(fee=0.001, slippage=0.0, threshold=0.02)

    analysis = analyze(RepeatLastReturn(), _closes(100, 101), HOUR_MS, settings)

    assert analysis.round_trip_cost == pytest.approx(0.002)
    assert analysis.expected_return_after_costs == pytest.approx(0.008)
    assert analysis.threshold == 0.02
    assert analysis.clears_entry_threshold is False


def test_analyze_flags_an_exit_on_a_fall() -> None:
    analysis = analyze(RepeatLastReturn(), _closes(100, 98), HOUR_MS, Settings())

    assert analysis.expected_return == pytest.approx(-0.02)
    assert analysis.expected_return_after_costs == pytest.approx(-0.027)
    assert analysis.clears_entry_threshold is False
    assert analysis.clears_exit_threshold is True


def test_hit_rate_scores_the_predicted_direction() -> None:
    analysis = analyze(RepeatLastReturn(), MIXED, HOUR_MS, Settings())

    assert analysis.hit_rate == pytest.approx(2 / 3)
    assert analysis.hit_rate_candles == 3


def test_hit_rate_covers_only_the_window() -> None:
    # The last 3 candles: 4 (a hit), 5 and 6 (skipped).
    analysis = analyze(RepeatLastReturn(), MIXED, HOUR_MS, Settings(hit_rate_window=3))

    assert analysis.hit_rate == 1.0
    assert analysis.hit_rate_candles == 1


def test_hit_rate_is_null_without_enough_history() -> None:
    # Two candles: enough to forecast, none with two candles before it.
    analysis = analyze(RepeatLastReturn(), _closes(100, 101), HOUR_MS, Settings())

    assert analysis.hit_rate is None
    assert analysis.hit_rate_candles == 0


def test_hit_rate_is_null_in_a_flat_market() -> None:
    analysis = analyze(
        RepeatLastReturn(), _closes(100, 100, 100, 100, 100), HOUR_MS, Settings()
    )

    assert analysis.hit_rate is None
    assert analysis.hit_rate_candles == 0


def test_hit_rate_only_scores_candles_after_the_training_data() -> None:
    trained = replace(RepeatLastReturn(), trained_through_ms=3 * HOUR_MS)

    analysis = analyze(trained, MIXED, HOUR_MS, Settings())

    # Candles 2 and 3 are in the training data; only 4 (a hit) is scored.
    assert analysis.hit_rate == 1.0
    assert analysis.hit_rate_candles == 1


def test_hit_rate_never_shows_the_forecaster_the_candle_it_scores() -> None:
    forecaster = RecordingForecaster()

    analyze(forecaster, _closes(100, 101, 102, 103), HOUR_MS, Settings())

    # The forecast itself (from 2 and 3), plus candles 2 and 3 each
    # predicted from the two before them.
    assert sorted(forecaster.windows) == [(0, 1), (1, 2), (2, 3)]


def test_analyze_raises_not_enough_candles() -> None:
    with pytest.raises(NotEnoughCandles):
        analyze(RepeatLastReturn(), _closes(100), HOUR_MS, Settings())


def test_analyze_is_deterministic() -> None:
    first = analyze(RepeatLastReturn(), MIXED, HOUR_MS, Settings())
    second = analyze(RepeatLastReturn(), MIXED, HOUR_MS, Settings())

    assert isinstance(first, Analysis)
    assert first == second


FACTORS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # signal, in thresholds


@pytest.mark.parametrize("factor", FACTORS)
def test_entry_flag_agrees_with_the_backtest(factor: float) -> None:
    settings = Settings()
    signal = factor * settings.round_trip_cost
    analysis = analyze(ConstantForecaster(signal), _closes(100, 101), HOUR_MS, settings)

    # The backtest opens a position on the first candle iff the signal
    # clears the threshold; an open position closes when the data ends.
    returns, _ = calculate_trade_returns(
        [0.0, 0.0],
        [signal, np.nan],
        cost_pct=settings.fee,
        slippage_pct=settings.slippage,
        threshold=settings.threshold,
    )

    assert analysis.clears_entry_threshold == (len(returns) == 1)


@pytest.mark.parametrize("factor", FACTORS)
def test_exit_flag_agrees_with_the_backtest(factor: float) -> None:
    settings = Settings()
    signal = factor * settings.round_trip_cost
    analysis = analyze(ConstantForecaster(signal), _closes(100, 101), HOUR_MS, settings)

    # Open on candle 0, then the signal on candle 1. A position still open
    # on candle 2 gains its +10%; one closed on candle 1 only pays costs.
    returns, _ = calculate_trade_returns(
        [0.0, 0.0, 0.1],
        [1.0, signal, np.nan],
        cost_pct=settings.fee,
        slippage_pct=settings.slippage,
        threshold=settings.threshold,
    )

    assert analysis.clears_exit_threshold == (returns[0] < 0)
