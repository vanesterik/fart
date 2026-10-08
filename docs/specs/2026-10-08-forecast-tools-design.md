# Forecast Tools — Design

**Date:** 2026-10-08
**Status:** Draft, awaiting review
**Source:** [Epic #45: Forecast Tools](https://github.com/vanesterik/fartt/issues/45), stories [#69](https://github.com/vanesterik/fartt/issues/69), [#74](https://github.com/vanesterik/fartt/issues/74), [#70](https://github.com/vanesterik/fartt/issues/70), [#71](https://github.com/vanesterik/fartt/issues/71), [#72](https://github.com/vanesterik/fartt/issues/72) and [#73](https://github.com/vanesterik/fartt/issues/73)
**Related PRD:** `docs/product/mcp-trading-agent-prd.md` (§5 Solution Overview, §7 epic C)
**Builds on:** `docs/specs/2026-10-07-server-foundation-design.md` (epic A: exchange layer, candle cache, MCP server)

## Problem

Epic A gave the agent market data: `get_candles`, served by `fartt serve` from `.mcp.json`. The next half of a trading cycle is a forecast and its after-cost numbers. Epic C adds `get_forecast`, `analyze_forecast` and `get_model_info`, backed at first by a naive baseline. The agent loop then runs end to end before epic B has selected a real model, and that model plugs in later without changing the tools.

The epic also turns the throwaway stdio checks used for #52 into a committed end-to-end test that every tool joins (#69). It renames the code's `Magnitude` to a return before the forecaster is built on it (#74).

## Decisions

- **Settings are `fartt serve` options with defaults** (`--fee`, `--slippage`, `--threshold`, `--hit-rate-window`). `.mcp.json` doesn't change unless the defaults should. They can move into the config file that epic D introduces for the risk limits, which the agent must not be able to change.
- **The forecast target stays the candle's simple return,** `close[t] / close[t-1] - 1` as a fraction (`0.004` = +0.4%). It's in the same units as profit and loss, fees and slippage, and `calculate_trade_returns` compounds it. Log returns differ negligibly at this scale (a 0.20% move is 0.1998% as a log return); a model that trains better on them can convert internally.
- **It's called a return, not a magnitude.** "Magnitude" normally means size without sign, while here the sign carries the direction. The tools say `expected_return`, and #74 renames the code (`Magnitude` → `Return`, `calculate_magnitude` → `calculate_returns`, and so on) right after #69, before #70 builds on it.
- **Approach A: a forecast package of plain functions, with thin tools in `server.py`.** The arithmetic is testable without MCP, and epic B's model implements the same Protocol. With four tools, `server.py` is about 300 lines. Epic D, which adds about eight tools, is the moment to decide whether to split the tools into modules grouped by data, forecast, portfolio and orders. The logic-in-packages part stays either way.
- **The baseline is "repeat the last return".** "No change" always forecasts 0 and would never give the loop a signal to act on. Both are the PRD's naive baselines, and beating them is the bar for epic B's models.

## Design

### 1. The forecast package (#70)

```
src/fartt/forecast/
├── __init__.py
├── forecaster.py   # Forecaster Protocol, Forecast, NotEnoughCandles, forecast()
├── baseline.py     # RepeatLastReturn
└── analysis.py     # Settings, Analysis, analyze()
```

- **`Forecaster`** is a `Protocol`, in the style of `ModelBuilder` and `Exchange`:
  - `name: str`, e.g. `"repeat-last-return"`
  - `kind: str`, e.g. `"naive baseline"` (epic B's model: `"trained model"`)
  - `required_candles: int`: how many closed candles it needs
  - `predict(candles: Sequence[Candle]) -> float`: the expected return of the candle after the last one given, as a fraction

  The model only predicts. Timestamps, freshness and costs are handled around it, so a trained model has one method to implement.
- **`forecast(forecaster, candles, interval_ms) -> Forecast`** wraps `predict`. It uses the last `required_candles` candles and returns a frozen `Forecast(model, expected_return, based_on_ms, applies_to_ms)`. `based_on_ms` is the newest candle used and `applies_to_ms` is one interval later. Fewer candles than required raises `NotEnoughCandles(needed, got)`.
- **`RepeatLastReturn`** needs 2 candles and predicts `close[-1] / close[-2] - 1`.
- **`src/fartt/model/predict_model.py`,** the empty stub, is removed; the forecaster replaces it.

### 2. The analysis (#72)

**`Settings`** is a frozen dataclass filled from `fartt serve`'s options:

| Option | Default | Meaning |
|---|---|---|
| `--fee` | `0.0025` | Fee per leg (Bitvavo's entry tier, 0.25% taker) |
| `--slippage` | `0.001` | Slippage per leg |
| `--threshold` | round-trip cost | Minimum expected return to act on |
| `--hit-rate-window` | `100` | Recent candles the hit rate covers |

**`analyze(forecaster, candles, interval_ms, settings) -> Analysis`** is a pure function of the candles and the settings. It returns:

- `expected_return`: the latest forecast.
- `round_trip_cost` = `2 × (fee + slippage)` (0.7% with the defaults).
- `expected_return_after_costs` = `expected_return − round_trip_cost`: the expected result of buying now and selling one candle later (long-only).
- `threshold`, plus two flags using the same rules as `calculate_trade_returns`, so the analysis and the backtest always agree:
  - `clears_entry_threshold`: `expected_return > threshold` (a long position would be worth opening);
  - `clears_exit_threshold`: `expected_return < −threshold` (an open position would be worth closing; used by epic D).
- `hit_rate` and `hit_rate_candles`. Over the last `hit_rate_window` closed candles, the forecaster predicts each candle from the `required_candles` before it. The hit rate is the fraction where the predicted direction matches the realised return; candles where either value is exactly 0 are skipped. `hit_rate_candles` is how many candles were scored. The window shortens with too little history, and `hit_rate` is `null` when nothing could be scored.

The analysis reports numbers and flags only. Deciding to trade is the agent's job, constrained in epic D by the server's risk limits.

### 3. The tools (#71, #72, #73)

`build_server(cache, market, interval, forecaster, settings)` gains the forecaster and the settings. `fartt serve` passes `RepeatLastReturn()`; there's no option to choose a model until epic B has one.

All three tools:

- take no arguments, since they work on the server's market and interval;
- only read the cache (`get_candles` updated it at the start of the cycle), and are annotated read-only and closed-world;
- follow `get_candles`' conventions: ISO 8601 UTC times, `is_current` plus a `warning` when the data is stale, and a `ToolError` that says what to do when they can't answer. `is_current` means the newest cached candle is the most recently closed period.

| Tool | Returns |
|---|---|
| `get_forecast` (#71) | `market`, `interval`, `model`, `expected_return`, `applies_to`, `based_on`, `is_current`, `warning` |
| `analyze_forecast` (#72) | `market`, `interval`, `model`, `applies_to`, `based_on`, `is_current`, `warning`, plus `expected_return`, `round_trip_cost`, `expected_return_after_costs`, `threshold`, `clears_entry_threshold`, `clears_exit_threshold`, `hit_rate`, `hit_rate_candles` |
| `get_model_info` (#73) | `name`, `kind`, `required_candles`, `trained_at`, `metrics`, `note`, `newest_cached`, `newest_closed`, `is_current` |

- For the baseline, `trained_at` and `metrics` are `null`, with a `note` that a naive baseline has no training or backtest and that epic B fills them in.
- Too little history is a `ToolError` naming how many candles are needed and how many are cached, and suggesting `fartt download` or a retry next cycle.
- The server's `instructions` describe the cycle order (`get_candles` → `get_forecast` → `analyze_forecast`) and `get_model_info` for judging trust.

### 4. The end-to-end check (#69)

`tests/e2e/test_server_stdio.py`, marked `network`, run by `task test:network`:

- **The server starts the way Claude Code starts it.** The test reads `command` and `args` from `.mcp.json` and appends `--assets-dir <temp dir> --interval 1d`; argparse takes the last value of a repeated option. The real registration is tested, and a fresh cache fills in three batches.
- **The trading cycle runs in one test, in order,** through `mcp.Client(StdioServerParameters(...))`. It checks that the server lists exactly the expected tools, then calls `get_candles` and checks its fields, that candles are returned, and `is_current: true`. Each tool story adds its call and its tool name to this test.
- **stdout carries only the protocol.** A second test starts the same command as a plain subprocess, sends `initialize`, `tools/list` and a `get_candles` call as JSON-RPC lines (with the SDK's current protocol version constant), and asserts every stdout line is a valid JSON-RPC message. The SDK's client may skip unparseable lines, so it can't catch this on its own.
- **It stays out of the default suite and the pre-push gate** (it calls Bitvavo). Each tool story runs it before committing.

## Testing

- **`fartt/forecast/`** (#70, #72): unit tests without MCP.
  - `RepeatLastReturn` on known closes, `forecast()` timestamps and `NotEnoughCandles`.
  - `analyze()` arithmetic pinned with hand-computed values, including the default threshold equal to the round-trip cost.
  - The hit rate: zeros skipped, a shortened window, `null` with no history.
  - Determinism: two calls give identical results.
  - Agreement with the backtest: for a sample signal, the threshold flags match `calculate_trade_returns`' entry and exit decisions with the same settings.
- **The tools** (#71–#73): the SDK's in-memory client against `build_server` with `FakeExchange`. Fields, stale data, too little history, read-only behaviour (the exchange is never called), and the input and output schemas.
- **`fartt serve`'s options** (#72): defaults, and nonsense values (a negative fee, a window of 0) rejected with exit code 2.
- **End to end** (#69): section 4, extended by each tool story.

## Delivery

One pull request per story, in the epic's order. This design lands in the first; later stories link to it on `main`.

| Pull request | Story | Commits |
|---|---|---|
| first | #69 end-to-end check | this design (with #69 and #74 added to the PRD's §7) → plan → task commits |
| next | #74 rename Magnitude → Return | plan → task commits |
| next | #70 forecaster and baseline | plan → task commits |
| next | #71 `get_forecast` | plan → task commits |
| next | #72 `analyze_forecast` and the four `serve` options | plan → task commits |
| next | #73 `get_model_info` | plan → task commits |

When #73 merges, the operator runs the acceptance script on epic #45.

## Out of scope

- Choosing or training a real model: epic B. `get_model_info`'s `trained_at` and `metrics` stay `null` until then.
- Portfolio, risk limits, orders and the decision journal: epic D, including the config file the settings may move into.
- Splitting `server.py` into per-area tool modules: decided in epic D's design.
- Forecasts more than one candle ahead.
