# Forecaster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the forecaster interface and its naive baseline, so the forecast tools can be built on it now and epic B's trained model can plug in later without changing them (story #70).

**Architecture:** A new `fartt.forecast` package. `forecaster.py` holds the `Forecaster` Protocol (read-only properties `name`, `kind`, `required_candles` and `trained_through_ms`, plus `predict`), the frozen `Forecast` result, `NotEnoughCandles`, and `forecast()`, which wraps `predict` with timestamps. `baseline.py` holds `RepeatLastReturn`. The empty `model/predict_model.py` stub is removed. Nothing calls the package yet; `get_forecast` (#71) is its first caller.

**Tech Stack:** Python 3.14, dataclasses, `typing.Protocol`, pytest, pyright (strict).

**Spec:** `docs/specs/2026-10-08-forecast-tools-design.md` (section "1. The forecast package")

## Global Constraints

- Returns are fractions (`0.004` = +0.4%), the candle's simple return `close[t] / close[t-1] - 1`, the same as `calculate_candle_returns`.
- The `Forecaster` Protocol declares its descriptive members as read-only `@property`s. Verified while writing this plan: under pyright strict, both a `@dataclass(frozen=True)` and a plain class satisfy such a Protocol.
- `predict(candles)` returns the expected return of the candle **after** the last one given. `forecast()` uses only the last `required_candles` candles.
- `trained_through_ms` is `None` for a model that isn't trained (the baseline); the hit rate (#72) uses it to stay out-of-sample.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **More candles than the forecaster needs** (the cache hands over 200): expect only the last `required_candles` to be used, and `based_on` to be the newest. Pinned by `test_forecast_uses_only_the_newest_required_candles`.
- **Exactly as many candles as needed:** expect a forecast, not an error (an off-by-one would raise). Pinned by `test_forecast_with_exactly_the_required_candles`.
- **No candles at all** (an empty cache): expect `NotEnoughCandles` with `got == 0`, not an `IndexError`. Pinned by `test_forecast_without_candles_raises_not_enough_candles`.
- **A falling price:** expect a negative return (the sign carries direction). Pinned by `test_repeat_last_return_is_negative_after_a_fall`.
- **Candles that aren't in time order:** they always are; the cache returns them sorted, oldest first. The baseline doesn't re-sort, and the docstring states the precondition.

---

## Task 1: The forecast package

**Files:**
- Create: `src/fartt/forecast/__init__.py`, `src/fartt/forecast/forecaster.py`, `src/fartt/forecast/baseline.py`
- Create: `tests/forecast/test_forecaster.py`, `tests/forecast/test_baseline.py`
- Delete: `src/fartt/model/predict_model.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Produces (used by #71–#73):
  - `fartt.forecast.Forecaster`: Protocol with `name -> str`, `kind -> str`, `required_candles -> int`, `trained_through_ms -> int | None` (read-only properties) and `predict(candles: Sequence[Candle]) -> float`
  - `fartt.forecast.Forecast(model: str, expected_return: float, based_on_ms: int, applies_to_ms: int)`, a frozen dataclass
  - `fartt.forecast.NotEnoughCandles(needed: int, got: int)`, an `Exception` with `.needed` and `.got`
  - `fartt.forecast.forecast(forecaster: Forecaster, candles: Sequence[Candle], interval_ms: int) -> Forecast`
  - `fartt.forecast.RepeatLastReturn()`, a frozen dataclass: `name="repeat-last-return"`, `kind="naive baseline"`, `required_candles=2`, `trained_through_ms=None`

- [ ] **Step 1: Write the failing tests**

Create `tests/forecast/test_baseline.py`:

```python
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
```

Create `tests/forecast/test_forecaster.py`:

```python
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
```

Run: `uv run pytest tests/forecast -q 2>&1 | grep -E "Error" | head -2`
Expected: `ModuleNotFoundError: No module named 'fartt.forecast'`.

- [ ] **Step 2: Implement the forecaster interface**

Create `src/fartt/forecast/forecaster.py`:

```python
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
```

(The message text `"needs 2 candles"`/`"got 1"` must match the test: `f"The forecaster needs {needed} candles but got {got}"`.)

- [ ] **Step 3: Implement the baseline and the package exports**

Create `src/fartt/forecast/baseline.py`:

```python
from collections.abc import Sequence
from dataclasses import dataclass

from fartt.exchange import Candle


@dataclass(frozen=True)
class RepeatLastReturn:
    """
    Naive baseline: forecast that the next candle repeats the last candle's
    return. One of the PRD's naive baselines -- every model epic B selects
    has to beat it. Chosen over "no change" because it gives the trading
    loop realistic, non-zero signals to act on.

    """

    name: str = "repeat-last-return"
    kind: str = "naive baseline"
    required_candles: int = 2
    trained_through_ms: int | None = None

    def predict(self, candles: Sequence[Candle]) -> float:
        previous, last = candles[-2], candles[-1]
        return last.close / previous.close - 1
```

Create `src/fartt/forecast/__init__.py`:

```python
from fartt.forecast.baseline import RepeatLastReturn
from fartt.forecast.forecaster import Forecast, Forecaster, NotEnoughCandles, forecast

__all__ = ["Forecast", "Forecaster", "NotEnoughCandles", "RepeatLastReturn", "forecast"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/forecast -q 2>&1 | tail -1`
Expected: `9 passed`.

- [ ] **Step 5: Remove the stub and update the docs**

Run: `git rm -q src/fartt/model/predict_model.py`

- `CLAUDE.md`:
  - In "What doesn't exist yet", the forecast-tools bullet becomes: "The forecast tools (`get_forecast`, `analyze_forecast`, `get_model_info`, epic C). Their forecaster exists: `fartt/forecast/` holds the `Forecaster` Protocol, `forecast()` and the `RepeatLastReturn` naive baseline; the tools come in #71–#73." Remove the sentence about `predict_model.py`.
  - Also remove `predict_model.py` wherever it's mentioned elsewhere (the "no `fartt train`" paragraph, and the Architecture `fartt/model/` bullet).
  - Add an Architecture bullet: "**`fartt/forecast/`** — the `Forecaster` Protocol (read-only `name`, `kind`, `required_candles`, `trained_through_ms`, plus `predict(candles) -> float`, the next candle's expected return as a fraction), the frozen `Forecast` (`model`, `expected_return`, `based_on_ms`, `applies_to_ms`), `NotEnoughCandles`, and `forecast()`, which wraps `predict` with timestamps. `RepeatLastReturn` is the naive baseline that epic B's trained model has to beat and will sit beside."
- `README.md`: in the structure tree, remove the `predict_model.py` line, and add after `exchange`:

```
        ├── forecast       <- The Forecaster interface and the naive baseline.
        │   ├── forecaster.py      <- Forecaster Protocol, Forecast, forecast().
        │   └── baseline.py        <- RepeatLastReturn.
        │
```

Run: `git grep -n "predict_model" -- src tests CLAUDE.md README.md`
Expected: no matches.

- [ ] **Step 6: Check, commit and push**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `147 passed, 3 deselected` (138 + 9).

```bash
git add -A src/fartt/forecast src/fartt/model/predict_model.py tests/forecast CLAUDE.md README.md
git commit -m "feat: add the forecaster interface and a naive baseline"
git push
```
