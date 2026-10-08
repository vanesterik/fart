# analyze_forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `analyze_forecast` tool: the latest forecast after trading costs, the entry and exit threshold flags, and the forecaster's recent out-of-sample hit rate. Also add the four `fartt serve` options that configure it (story #72).

**Architecture:**
- Task 1 adds `fartt/forecast/analysis.py`: a frozen `Settings`, a frozen `Analysis`, and `analyze()`, a pure function of the candles and the settings. Its threshold flags follow `calculate_trade_returns`' rules, so the analysis and the backtest agree.
- Task 2 wires it in:
  - `build_server` gains `settings: Settings`;
  - the new tool shares `get_forecast`'s cache read, its too-little-history error and its staleness check, which move into helpers;
  - `fartt serve` gains `--fee`, `--slippage`, `--threshold` and `--hit-rate-window`, validated at startup;
  - the end-to-end check, `CLAUDE.md` and the README follow.

**Tech Stack:** MCP Python SDK v2, Pydantic, argparse, pytest with anyio, pyright (strict).

**Spec:** `docs/specs/2026-10-08-forecast-tools-design.md` (Decisions on units and settings, "2. The analysis", "3. The tools", Testing)

**Where the settings live (the story's open question):** `fartt serve` options with defaults, as the spec decides. `.mcp.json` doesn't change. They can move into epic D's config file later.

## Global Constraints

- **Units:** every return, cost, threshold and rate is a fraction (`0.007` = 0.7%; a hit rate of `0.55` = 55%), in tool output and in the options. Field descriptions and option help say so.
- **Defaults:**
  - `--fee 0.0025` per leg;
  - `--slippage 0.001` per leg;
  - `--threshold` defaults to the round-trip cost, `2 * (fee + slippage)` (`0.007` with the defaults);
  - `--hit-rate-window 100`.
- **Formulas:**
  - `expected_return_after_costs = expected_return - round_trip_cost`;
  - `clears_entry_threshold = expected_return > threshold`;
  - `clears_exit_threshold = expected_return < -threshold`;
  - all are the same rules as `calculate_trade_returns`.
- **Validation** through `parser.error` (exit code 2):
  - `--fee` and `--slippage` must be finite and `>= 0`;
  - `--threshold`, if given, must be finite and `>= 0`;
  - `--hit-rate-window` must be `>= 1`;
  - a threshold below the round-trip cost is allowed but logged as a warning at startup.
- **Hit rate:**
  - **What's scored:** over the last `hit_rate_window` closed candles, each is predicted from the `required_candles` before it, never seeing itself.
  - **What's skipped:** a candle is skipped when the prediction or the realised return is exactly 0, or when it opens at or before `forecaster.trained_through_ms`.
  - **Output:** `hit_rate_candles` counts the scored candles, and `hit_rate` is `null` when that count is 0.
  - **Short history:** with too little history, the window shortens.
- **The tool:**
  - it takes no arguments, is read-only and closed-world, and never contacts the exchange;
  - its timing fields, `is_current`/`warning` and too-little-history `ToolError` are exactly `get_forecast`'s.
- **Signatures:** `build_server(cache, market, interval, forecaster, settings)`, with `settings` required, like `forecaster`.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **An operator types a percentage as if it were a fraction** (`--fee 0.25` meaning 0.25%): the option is valid but means 25%. Expect the help to say "as a fraction" with an example, and the startup log to print the settings as percentages ("fee 25.00% per leg"), so the mistake is visible. Pinned by `test_serve_runs_the_server_for_the_configured_market`, which checks the log line for the defaults.
- **`nan` or `inf` as an option value** (argparse's `float` accepts both): `nan >= 0` is false and `inf >= 0` is true, so a plain comparison would let one of them through. Expect exit code 2 for both. Pinned by `test_serve_rejects_nonsense_analysis_options`.
- **A flat market,** where every return is 0: expect `hit_rate: null` and `hit_rate_candles: 0`, not a division by zero. Pinned by `test_hit_rate_is_null_in_a_flat_market`.
- **A freshly trained model,** whose training data covers most of the window: expect only the later candles to be scored, and `null` if there are none. Pinned by `test_hit_rate_only_scores_candles_after_the_training_data`.
- **A very large `--hit-rate-window`** (e.g. 100000): the tool reads `required_candles + window` candles from the cache tail on every call. Expect it to be slow rather than wrong. Not tested: it's a cost, not a correctness issue, and the default is 100.

---

## Task 1: The analysis

**Files:**
- Create: `src/fartt/forecast/analysis.py`, `tests/forecast/test_analysis.py`
- Modify: `src/fartt/forecast/__init__.py`

**Interfaces:**
- Consumes: `Forecaster`, `forecast()`, `NotEnoughCandles` (`fartt.forecast.forecaster`); `RepeatLastReturn`; `calculate_trade_returns` (tests only).
- Produces:
  - `Settings(fee: float = 0.0025, slippage: float = 0.001, threshold: float | None = None, hit_rate_window: int = 100)`, with the property `round_trip_cost -> float`;
  - `Analysis(model, expected_return, based_on_ms, applies_to_ms, round_trip_cost, expected_return_after_costs, threshold, clears_entry_threshold, clears_exit_threshold, hit_rate: float | None, hit_rate_candles: int)`;
  - `analyze(forecaster: Forecaster, candles: Sequence[Candle], interval_ms: int, settings: Settings) -> Analysis`. It raises `NotEnoughCandles` like `forecast()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/forecast/test_analysis.py`:

```python
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from fartt.exchange import Candle
from fartt.features.calculate_trade_returns import calculate_trade_returns
from fartt.forecast import Analysis, NotEnoughCandles, RepeatLastReturn, Settings, analyze
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
```

Run: `uv run pytest tests/forecast/test_analysis.py -q 2>&1 | tail -3`
Expected: a collection error, `ImportError: cannot import name 'Analysis' from 'fartt.forecast'`.

- [ ] **Step 2: Write the analysis**

Create `src/fartt/forecast/analysis.py`:

```python
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
```

`src/fartt/forecast/__init__.py` becomes:

```python
from fartt.forecast.analysis import Analysis, Settings, analyze
from fartt.forecast.baseline import RepeatLastReturn
from fartt.forecast.forecaster import Forecast, Forecaster, NotEnoughCandles, forecast

__all__ = [
    "Analysis",
    "Forecast",
    "Forecaster",
    "NotEnoughCandles",
    "RepeatLastReturn",
    "Settings",
    "analyze",
    "forecast",
]
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/forecast/test_analysis.py -q 2>&1 | tail -1`
Expected: `25 passed` (11 tests, plus 2 parametrized tests × 7 factors).

- [ ] **Step 4: Check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `180 passed, 3 deselected`.

```bash
git add src/fartt/forecast tests/forecast/test_analysis.py
git commit -m "feat: add the after-cost analysis with an out-of-sample hit rate"
git push
```

---

## Task 2: The `analyze_forecast` tool and the `serve` options

**Files** (8):
- Modify: `src/fartt/server/server.py`, `src/fartt/server/__init__.py`, `src/fartt/cli.py`
- Modify: `tests/test_server.py`, `tests/test_cli.py`, `tests/e2e/test_server_stdio.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `Settings`, `Analysis`, `analyze()` (Task 1); `build_server`'s `get_forecast` (#71).
- Produces:
  - `build_server(cache: CandleCache, market: str, interval: str, forecaster: Forecaster, settings: Settings) -> MCPServer`;
  - the tool `analyze_forecast() -> AnalysisResult`;
  - `fartt serve --fee --slippage --threshold --hit-rate-window`.

- [ ] **Step 1: Write the failing server tests**

In `tests/test_server.py`:
- change the import to `from fartt.forecast import RepeatLastReturn, Settings`;
- replace the helper `_server` with the version below, and add `_analyze_forecast` below `_get_forecast`.

```python
def _server(cache: CandleCache, settings: Settings | None = None) -> Any:
    return build_server(
        cache, "BTC/EUR", "1h", RepeatLastReturn(), settings or Settings()
    )


async def _analyze_forecast(cache: CandleCache, settings: Settings | None = None) -> Any:
    async with Client(_server(cache, settings)) as client:
        return await client.call_tool("analyze_forecast", {})
```

In the tool-list test (`test_server_offers_get_candles_as_a_read_only_tool`):
- change the names assertion to `["get_candles", "get_forecast", "analyze_forecast"]`;
- append:

```python
    analysis_tool = tools[2]
    assert analysis_tool.annotations is not None
    assert analysis_tool.annotations.read_only_hint is True
    assert analysis_tool.annotations.open_world_hint is False
    assert analysis_tool.input_schema.get("properties", {}) == {}
    assert analysis_tool.output_schema is not None
    hit_rate = analysis_tool.output_schema["properties"]["hit_rate"]
    assert "fraction" in hit_rate["description"]
```

Append the new tests:

```python
@pytest.mark.anyio
async def test_analyze_forecast_returns_the_after_cost_analysis(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange(
        [candle(0, close=100.0), candle(1, close=101.0), candle(2, close=102.0)],
        now_ms=3 * HOUR_MS,
    )
    cache = _cache(exchange, tmp_path)
    cache.update()

    result = await _analyze_forecast(cache)

    assert not result.is_error, result.content
    expected_return = 102 / 101 - 1
    assert result.structured_content == {
        "market": "BTC/EUR",
        "interval": "1h",
        "model": "repeat-last-return",
        "applies_to": "1970-01-01T03:00:00+00:00",
        "based_on": "1970-01-01T02:00:00+00:00",
        "is_current": True,
        "warning": None,
        "expected_return": pytest.approx(expected_return),
        "round_trip_cost": pytest.approx(0.007),
        "expected_return_after_costs": pytest.approx(expected_return - 0.007),
        "threshold": pytest.approx(0.007),
        "clears_entry_threshold": True,
        "clears_exit_threshold": False,
        "hit_rate": 1.0,  # candle 2 rose, as candle 1 did
        "hit_rate_candles": 1,
    }


@pytest.mark.anyio
async def test_analyze_forecast_uses_the_servers_settings(tmp_path: Path) -> None:
    exchange = FakeExchange(
        [candle(0, close=100.0), candle(1, close=101.0)], now_ms=2 * HOUR_MS
    )
    cache = _cache(exchange, tmp_path)
    cache.update()

    settings = Settings(fee=0.0, slippage=0.0, threshold=0.02)
    content = (await _analyze_forecast(cache, settings)).structured_content

    assert content["round_trip_cost"] == 0.0
    assert content["expected_return_after_costs"] == pytest.approx(0.01)
    assert content["threshold"] == 0.02
    assert content["clears_entry_threshold"] is False


@pytest.mark.anyio
async def test_analyze_forecast_flags_a_stale_cache(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.now_ms = 5 * HOUR_MS  # two more periods closed; cache not updated

    content = (await _analyze_forecast(cache)).structured_content

    assert content["is_current"] is False
    assert "get_candles" in content["warning"]


@pytest.mark.anyio
async def test_analyze_forecast_explains_too_little_history(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=3 * HOUR_MS), tmp_path)

    result = await _analyze_forecast(cache)

    assert result.is_error
    text = result.content[0].text
    assert "needs 2" in text
    assert "holds 0" in text
    assert "uv run fartt download --market BTC/EUR --interval 1h" in text


@pytest.mark.anyio
async def test_analyze_forecast_never_calls_the_exchange(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.fetch_calls.clear()
    exchange.reachable = False  # it would raise if called

    result = await _analyze_forecast(cache)

    assert not result.is_error, result.content
    assert exchange.fetch_calls == []
```

`test_analyze_forecast_returns_the_after_cost_analysis` also proves that the tool reads more than `required_candles`: with only the 2 newest candles, `hit_rate` would be `None`.

- [ ] **Step 2: Write the failing CLI tests**

In `tests/test_cli.py`:
- import `Settings` alongside `RepeatLastReturn`;
- make the `served` fixture's fake accept and record `settings` (a fifth parameter, `settings: Any`, and `settings=settings` in `built.update`);
- in `test_serve_runs_the_server_for_the_configured_market`, add the `capsys: pytest.CaptureFixture[str]` parameter and these assertions:

```python
    assert served["settings"] == Settings()
    assert "fee 0.25% per leg" in capsys.readouterr().err
```

Append:

```python
def test_serve_passes_the_analysis_options(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    cli.main(
        [
            "serve",
            "--assets-dir",
            str(tmp_path),
            "--interval",
            "1h",
            "--fee",
            "0.001",
            "--slippage",
            "0",
            "--threshold",
            "0.01",
            "--hit-rate-window",
            "50",
        ]
    )

    assert served["settings"] == Settings(
        fee=0.001, slippage=0.0, threshold=0.01, hit_rate_window=50
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--fee", "-0.001"),
        ("--fee", "nan"),
        ("--slippage", "inf"),
        ("--threshold", "-0.01"),
        ("--hit-rate-window", "0"),
    ],
)
def test_serve_rejects_nonsense_analysis_options(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--assets-dir", str(tmp_path), option, value])

    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err
    assert not served


def test_serve_warns_about_a_threshold_below_the_round_trip_cost(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["serve", "--assets-dir", str(tmp_path), "--threshold", "0.001"])

    assert "below the round-trip cost" in capsys.readouterr().err
    assert served["server"].ran
```

Run: `uv run pytest tests/test_server.py tests/test_cli.py -q 2>&1 | tail -1`
Expected: failures. `build_server` doesn't take settings, `analyze_forecast` doesn't exist, and `serve` doesn't know the options.

- [ ] **Step 3: Add the tool**

In `src/fartt/server/server.py`:
- import `from fartt.forecast import Forecaster, NotEnoughCandles, Settings, analyze, forecast`;
- replace `INSTRUCTIONS` with:

```python
INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Every trading cycle starts with "
    "get_candles, which updates the cache from the exchange, then "
    "get_forecast for the next candle's expected return, and "
    "analyze_forecast for that return after trading costs, whether it clears "
    "the entry or exit threshold, and the model's recent hit rate. Returns, "
    "costs and rates are fractions: 0.004 means 0.4%. If is_current is false "
    "the data is stale: read the warning and treat the cycle as a hold."
)

STALE_FORECAST_WARNING = (
    "The newest cached candle is older than the most recently closed period, "
    "so this forecast may be stale. Call get_candles first to update the "
    "cache; if it also reports is_current: false, treat this cycle as a hold."
)
```

Add the result model below `ForecastResult`:

```python
class AnalysisResult(BaseModel):
    market: str
    interval: str
    model: str = Field(description="Name of the model that made the forecast.")
    applies_to: str = Field(
        description="Open time of the candle being forecast, ISO 8601 in UTC."
    )
    based_on: str = Field(
        description="Open time of the newest candle the forecast used, ISO 8601 in UTC."
    )
    is_current: bool = Field(
        description=(
            "True when the forecast is based on the most recently closed "
            "period. False means the cache is behind; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="Why the forecast may be stale, if it may be."
    )
    expected_return: float = Field(
        description="Expected return of the next candle, as a fraction: 0.004 means +0.4%."
    )
    round_trip_cost: float = Field(
        description="Fees and slippage of buying and later selling, as a fraction."
    )
    expected_return_after_costs: float = Field(
        description=(
            "expected_return minus round_trip_cost: the expected result of "
            "buying now and selling one candle later, as a fraction."
        )
    )
    threshold: float = Field(
        description="Minimum expected return to act on, as a fraction."
    )
    clears_entry_threshold: bool = Field(
        description="expected_return > threshold: a long position would be worth opening."
    )
    clears_exit_threshold: bool = Field(
        description="expected_return < -threshold: an open position would be worth closing."
    )
    hit_rate: float | None = Field(
        description=(
            "Fraction of recent candles whose direction the model predicted "
            "correctly, out-of-sample; null when none could be scored."
        )
    )
    hit_rate_candles: int = Field(
        description="How many recent candles the hit rate scored."
    )
```

Change the signature to `def build_server(cache: CandleCache, market: str, interval: str, forecaster: Forecaster, settings: Settings) -> MCPServer:`.

Below `server = MCPServer(...)`, add the helpers the two forecast tools share:

```python
    def read_candles(tool: str, count: int) -> list[Candle]:
        try:
            return cache.latest(count)
        except OSError as error:
            logger.error(f"{tool}: {error}")
            raise ToolError(
                f"Couldn't read the candle cache at {cache.filepath}: {error}. "
                "Retrying won't help until the operator fixes the file or its "
                "folder."
            ) from error

    def too_little_history(error: NotEnoughCandles) -> ToolError:
        return ToolError(
            f"{forecaster.name} needs {error.needed} closed {market} "
            f"{interval} candles and the cache holds {error.got}. Call "
            "get_candles to update it, or have the operator fill it with "
            f"`uv run fartt download --market {market} --interval {interval}`; "
            "otherwise retry next cycle."
        )
```

Rewrite `get_forecast`'s body to use them. Its docstring, annotations and return value stay the same:

```python
        candles = read_candles("get_forecast", forecaster.required_candles)
        try:
            result = forecast(forecaster, candles, interval_ms)
        except NotEnoughCandles as error:
            raise too_little_history(error) from error

        is_current = result.based_on_ms == cache.newest_closed_ms()
        warning = None if is_current else STALE_FORECAST_WARNING
```

(the `logger.info` and `return ForecastResult(...)` lines follow unchanged).

Add the tool after `get_forecast`, before `return server`:

```python
    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def analyze_forecast() -> AnalysisResult:
        """
        Analyse the next candle's forecast after trading costs: the expected
        return after a round trip's fees and slippage, whether it clears the
        entry or exit threshold, and the model's recent directional hit rate.
        Call get_candles first: this tool reads the cache and never contacts
        the exchange.
        """
        candles = read_candles(
            "analyze_forecast", forecaster.required_candles + settings.hit_rate_window
        )
        try:
            analysis = analyze(forecaster, candles, interval_ms, settings)
        except NotEnoughCandles as error:
            raise too_little_history(error) from error

        is_current = analysis.based_on_ms == cache.newest_closed_ms()
        hit_rate = "n/a" if analysis.hit_rate is None else f"{analysis.hit_rate:.0%}"
        logger.info(
            f"analyze_forecast: {analysis.model} expects "
            f"{analysis.expected_return:+.4%}, "
            f"{analysis.expected_return_after_costs:+.4%} after costs, "
            f"entry={analysis.clears_entry_threshold} "
            f"exit={analysis.clears_exit_threshold}, hit rate {hit_rate} over "
            f"{analysis.hit_rate_candles} candles, current={is_current}"
        )
        return AnalysisResult(
            market=market,
            interval=interval,
            model=analysis.model,
            applies_to=_iso(analysis.applies_to_ms),
            based_on=_iso(analysis.based_on_ms),
            is_current=is_current,
            warning=None if is_current else STALE_FORECAST_WARNING,
            expected_return=analysis.expected_return,
            round_trip_cost=analysis.round_trip_cost,
            expected_return_after_costs=analysis.expected_return_after_costs,
            threshold=analysis.threshold,
            clears_entry_threshold=analysis.clears_entry_threshold,
            clears_exit_threshold=analysis.clears_exit_threshold,
            hit_rate=analysis.hit_rate,
            hit_rate_candles=analysis.hit_rate_candles,
        )
```

`src/fartt/server/__init__.py` becomes:

```python
from fartt.server.server import (
    AnalysisResult,
    CandlesResult,
    ForecastResult,
    build_server,
)

__all__ = ["AnalysisResult", "CandlesResult", "ForecastResult", "build_server"]
```

- [ ] **Step 4: Add the `serve` options**

In `src/fartt/cli.py`:
- import `math`;
- import `from fartt.forecast import RepeatLastReturn, Settings`;
- add below `_add_cache_options`:

```python
def _add_analysis_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--fee",
        type=float,
        default=Settings.fee,
        help="trading fee per leg, as a fraction: 0.0025 is 0.25%% (default: %(default)s)",
    )
    command.add_argument(
        "--slippage",
        type=float,
        default=Settings.slippage,
        help="slippage per leg, as a fraction (default: %(default)s)",
    )
    command.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="minimum expected return to act on, as a fraction "
        "(default: the round-trip cost, 2 * (fee + slippage))",
    )
    command.add_argument(
        "--hit-rate-window",
        type=int,
        default=Settings.hit_rate_window,
        help="recent candles the hit rate covers (default: %(default)s)",
    )
```

In `build_parser`, call `_add_analysis_options(serve)` after `_add_cache_options(serve, interval="1h")`.

Add below `_open_cache`:

```python
def _settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Settings:
    # argparse's float accepts "nan" and "inf", so check finiteness too.
    for option, value in (("--fee", args.fee), ("--slippage", args.slippage)):
        if not (math.isfinite(value) and value >= 0):
            parser.error(f"{option} must be a fraction >= 0, e.g. 0.0025 for 0.25%")
    if args.threshold is not None and not (
        math.isfinite(args.threshold) and args.threshold >= 0
    ):
        parser.error("--threshold must be a fraction >= 0, e.g. 0.007 for 0.7%")
    if args.hit_rate_window < 1:
        parser.error("--hit-rate-window must be at least 1")

    settings = Settings(
        fee=args.fee,
        slippage=args.slippage,
        threshold=args.threshold,
        hit_rate_window=args.hit_rate_window,
    )
    if args.threshold is not None and args.threshold < settings.round_trip_cost:
        logger.warning(
            f"--threshold {args.threshold:.2%} is below the round-trip cost "
            f"{settings.round_trip_cost:.2%}: entries can lose money on costs alone"
        )
    return settings
```

In `main`, the serve branch becomes:

```python
    elif args.command == "serve":
        settings = _settings(parser, args)
        _serve(parser, args.assets_dir, args.exchange, args.market, args.interval, settings)
```

`_serve` takes `settings: Settings` as its last parameter. Before `build_server`, it logs the settings as percentages, so an operator who typed a percentage for a fraction sees it:

```python
    threshold = settings.round_trip_cost if settings.threshold is None else settings.threshold
    logger.info(
        f"Analysis settings: fee {settings.fee:.2%} per leg, slippage "
        f"{settings.slippage:.2%} per leg, threshold {threshold:.2%}, "
        f"hit rate over {settings.hit_rate_window} candles"
    )
    build_server(cache, market, interval, RepeatLastReturn(), settings).run()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_server.py tests/test_cli.py -q 2>&1 | tail -1`
Expected: `41 passed` (23 server: 18 + 5; 18 CLI: 11 + 1 + 5 + 1).

- [ ] **Step 6: Add `analyze_forecast` to the end-to-end check**

In `tests/e2e/test_server_stdio.py`:
- set `EXPECTED_TOOLS = ["get_candles", "get_forecast", "analyze_forecast"]`;
- append to `test_trading_cycle_over_stdio`, after the `get_forecast` checks:

```python
        analysis = await client.call_tool("analyze_forecast", {})
        assert not analysis.is_error, analysis.content
        numbers = analysis.structured_content
        assert numbers["applies_to"] == prediction["applies_to"]
        assert numbers["expected_return"] == prediction["expected_return"]
        assert numbers["round_trip_cost"] == pytest.approx(0.007)
        assert numbers["threshold"] == pytest.approx(0.007)
        assert 0 < numbers["hit_rate_candles"] <= 100
        assert 0 <= numbers["hit_rate"] <= 1
        assert numbers["is_current"] is True
```

Run: `task test:network 2>&1 | grep -E "passed|failed"`
Expected: `3 passed`.

- [ ] **Step 7: Update the docs**

- **`CLAUDE.md`:**
  - **Current-state server bullet:** three tools so far, `get_candles`, `get_forecast` and `analyze_forecast`.
  - **"What doesn't exist yet":** the bullet becomes "`get_model_info` (#73). `get_forecast` and `analyze_forecast` exist, backed by the `RepeatLastReturn` baseline from `fartt/forecast/`."
  - **The `serve` paragraph:** add that `serve` also takes `--fee` (0.0025), `--slippage` (0.001), `--threshold` (default the round-trip cost) and `--hit-rate-window` (100). All are fractions except the window, validated with exit code 2, with a startup warning for a threshold below the round-trip cost.
  - **The `fartt/forecast/` Architecture bullet:** add `analysis.py`: `Settings` (with `round_trip_cost`), `Analysis` and `analyze()`, whose threshold flags follow `calculate_trade_returns`' rules, and whose hit rate scores only candles after `trained_through_ms`.
  - **The `fartt/server/` bullet:** `build_server(cache, market, interval, forecaster, settings)`, and `analyze_forecast()`, which returns `get_forecast`'s timing fields plus `expected_return`, `round_trip_cost`, `expected_return_after_costs`, `threshold`, the two threshold flags, `hit_rate` and `hit_rate_candles`.
- **`README.md`:**
  - "What works today" adds the after-cost analysis through `analyze_forecast`;
  - "Today `get_candles` and `get_forecast` exist" becomes "Today `get_candles`, `get_forecast` and `analyze_forecast` exist";
  - the diagram's `analyze_forecast (planned)` loses "(planned)";
  - the tree gets `analysis.py <- Settings, Analysis, analyze().` under `forecast`, and the server line names the three tools;
  - after the `fartt download` example in Usage, add one sentence: "`fartt serve` also takes the analysis settings: `--fee`, `--slippage`, `--threshold` and `--hit-rate-window`, all fractions except the window (see `uv run fartt serve --help`)."

- [ ] **Step 8: Check, commit and push**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `192 passed, 3 deselected`.

```bash
git add src/fartt/server src/fartt/cli.py tests/test_server.py tests/test_cli.py tests/e2e/test_server_stdio.py CLAUDE.md README.md
git commit -m "feat: add the analyze_forecast tool and the serve analysis options"
git push
```
