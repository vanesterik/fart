# get_model_info Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `get_model_info` tool, which describes the model behind the forecasts and how fresh the cached data is, so the agent can judge how far to trust a forecast (story #73, the last of epic C).

**Architecture:** One thin tool in `server.py`, next to the other three. It reads only the newest cached candle and the cache's clock, and describes the forecaster through its existing Protocol members. The end-to-end check, `CLAUDE.md` and the README follow. The Protocol is unchanged.

**Tech Stack:** MCP Python SDK v2, Pydantic, pytest with anyio, pyright (strict).

**Spec:** `docs/specs/2026-10-08-forecast-tools-design.md` ("3. The tools", Out of scope)

## How `trained_at` and `metrics` map (decided here, as the design asked)

- **`trained_at`** is the date the model was trained. The `Forecaster` Protocol has no member for it, and `trained_through_ms` is not it: that is the **data cutoff**, the open time of the last training candle.
  - The tool returns `trained_at: null` for every forecaster until epic B adds a member such as `trained_at_ms` along with its trained model.
  - Adding it to the Protocol now would only give the baseline another `None` (see the design preferences in `CLAUDE.md`).
- **`metrics`** are the screening and backtest metrics, typed `dict[str, float] | None`. The tool returns `null` for the same reason, until epic B adds a Protocol member for them.
- **`trained_through`** is a field the design's table doesn't list; this plan adds it.
  - It is the ISO form of `trained_through_ms`, or `null` for the baseline.
  - It's the one training fact the Protocol already has, and it tells the agent where `analyze_forecast`'s out-of-sample hit rate starts.
- **`market` and `interval`** are added too, as every other tool returns them.
- **`note`** always carries the sentence about the missing training date and metrics. It gains a second sentence when the cache is empty or stale, saying what to do.

## Global Constraints

- **No arguments.** The tool is read-only and closed-world, and it never contacts the exchange.
- **It always answers,** even with an empty cache: `newest_cached` is then `null`, `is_current` is `false`, and the `note` says the cache is empty, giving `get_candles` and the exact `uv run fartt download --market <market> --interval <interval>` command.
- **The one error:** an unreadable cache file is a `ToolError`, the shared `read_candles` helper's.
- **`is_current`** means the newest cached candle is the most recently closed period (`cache.newest_closed_ms()`), the same rule as the other tools.
- **Times are candle open times,** ISO 8601 in UTC, and the field descriptions say so.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **A fresh install** (empty cache, before any download): expect an answer, not an error, with `newest_cached: null`, `is_current: false` and a note giving the exact download command. Pinned by `test_get_model_info_answers_with_an_empty_cache`.
- **The agent asks before `get_candles` in a new cycle:** expect `is_current: false` and a note to call `get_candles`. Pinned by `test_get_model_info_flags_stale_data`.
- **An agent reading `trained_through` as the training date:** expect the field description to say it's the training data's last candle. Expect `trained_at` to be a separate field, `null` until epic B. Pinned by the exact-dict test (both fields) and the tool-list test (the description).
- **A trained forecaster** (epic B, simulated by `replace(RepeatLastReturn(), trained_through_ms=...)`): expect `trained_through` set and `trained_at`/`metrics` still `null`, without crashing. Pinned by `test_get_model_info_reports_the_training_data_cutoff`.
- **The exchange is down:** expect the tool to work anyway. Pinned by `test_get_model_info_never_calls_the_exchange`.

---

## Task 1: The `get_model_info` tool

**Files** (6):
- Modify: `src/fartt/server/server.py`, `src/fartt/server/__init__.py`
- Modify: `tests/test_server.py`, `tests/e2e/test_server_stdio.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `Forecaster` (`name`, `kind`, `required_candles`, `trained_through_ms`); `CandleCache.newest_closed_ms()`; `build_server`'s `read_candles(tool, count)` helper and `_iso()` (#72).
- Produces: the tool `get_model_info() -> ModelInfoResult` (`market`, `interval`, `name`, `kind`, `required_candles`, `trained_through`, `trained_at`, `metrics`, `note`, `newest_cached`, `newest_closed`, `is_current`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_server.py`:
- add `from dataclasses import replace` at the top;
- change the import to `from fartt.forecast import Forecaster, RepeatLastReturn, Settings`;
- replace `_server` with:

```python
def _server(
    cache: CandleCache,
    settings: Settings | None = None,
    forecaster: Forecaster | None = None,
) -> Any:
    return build_server(
        cache,
        "BTC/EUR",
        "1h",
        forecaster or RepeatLastReturn(),
        settings or Settings(),
    )
```

and add below `_analyze_forecast`:

```python
async def _get_model_info(
    cache: CandleCache, forecaster: Forecaster | None = None
) -> Any:
    async with Client(_server(cache, forecaster=forecaster)) as client:
        return await client.call_tool("get_model_info", {})
```

In the tool-list test:
- the names assertion becomes `["get_candles", "get_forecast", "analyze_forecast", "get_model_info"]`;
- append:

```python
    info_tool = tools[3]
    assert info_tool.annotations is not None
    assert info_tool.annotations.read_only_hint is True
    assert info_tool.annotations.open_world_hint is False
    assert info_tool.input_schema.get("properties", {}) == {}
    assert info_tool.output_schema is not None
    trained_through = info_tool.output_schema["properties"]["trained_through"]
    assert "last candle in the training data" in trained_through["description"]
```

Append the new tests:

```python
@pytest.mark.anyio
async def test_get_model_info_describes_the_baseline(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()

    result = await _get_model_info(cache)

    assert not result.is_error, result.content
    content = dict(result.structured_content)
    note = content.pop("note")
    assert content == {
        "market": "BTC/EUR",
        "interval": "1h",
        "name": "repeat-last-return",
        "kind": "naive baseline",
        "required_candles": 2,
        "trained_through": None,
        "trained_at": None,
        "metrics": None,
        "newest_cached": "1970-01-01T02:00:00+00:00",
        "newest_closed": "1970-01-01T02:00:00+00:00",
        "is_current": True,
    }
    assert "no training date" in note
    assert "trained model" in note


@pytest.mark.anyio
async def test_get_model_info_answers_with_an_empty_cache(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=3 * HOUR_MS), tmp_path)

    result = await _get_model_info(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content["newest_cached"] is None
    assert content["newest_closed"] == "1970-01-01T02:00:00+00:00"
    assert content["is_current"] is False
    assert "empty" in content["note"]
    assert "uv run fartt download --market BTC/EUR --interval 1h" in content["note"]


@pytest.mark.anyio
async def test_get_model_info_flags_stale_data(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.now_ms = 5 * HOUR_MS  # two more periods closed; cache not updated

    content = (await _get_model_info(cache)).structured_content

    assert content["newest_cached"] == "1970-01-01T02:00:00+00:00"
    assert content["newest_closed"] == "1970-01-01T04:00:00+00:00"
    assert content["is_current"] is False
    assert "get_candles" in content["note"]


@pytest.mark.anyio
async def test_get_model_info_reports_the_training_data_cutoff(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    trained = replace(RepeatLastReturn(), trained_through_ms=HOUR_MS)

    content = (await _get_model_info(cache, trained)).structured_content

    assert content["trained_through"] == "1970-01-01T01:00:00+00:00"
    assert content["trained_at"] is None  # the cutoff is not the training date
    assert content["metrics"] is None


@pytest.mark.anyio
async def test_get_model_info_never_calls_the_exchange(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.fetch_calls.clear()
    exchange.reachable = False  # it would raise if called

    result = await _get_model_info(cache)

    assert not result.is_error, result.content
    assert exchange.fetch_calls == []
```

Run: `uv run pytest tests/test_server.py -q 2>&1 | tail -1`
Expected: 6 failures (the tool-list test and the 5 new tests: `get_model_info` doesn't exist), 24 passed.

- [ ] **Step 2: Add the tool**

In `src/fartt/server/server.py`, replace `INSTRUCTIONS` with:

```python
INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Every trading cycle starts with "
    "get_candles, which updates the cache from the exchange, then "
    "get_forecast for the next candle's expected return, and "
    "analyze_forecast for that return after trading costs, whether it clears "
    "the entry or exit threshold, and the model's recent hit rate. Call "
    "get_model_info to judge how far to trust the forecasts: which model "
    "makes them, its training and metrics, and how fresh the data is. "
    "Returns, costs and rates are fractions: 0.004 means 0.4%. If is_current "
    "is false the data is stale: read the warning and treat the cycle as a "
    "hold."
)
```

Add the result model below `AnalysisResult`:

```python
class ModelInfoResult(BaseModel):
    market: str
    interval: str
    name: str = Field(description="Name of the model behind the forecasts.")
    kind: str = Field(
        description='What sort of model, e.g. "naive baseline" or "trained model".'
    )
    required_candles: int = Field(
        description="Closed candles the model reads for one forecast."
    )
    trained_through: str | None = Field(
        description=(
            "Open time of the last candle in the training data, ISO 8601 in "
            "UTC: the data cutoff, not the training date. analyze_forecast's "
            "hit rate only scores later candles. Null for an untrained model."
        )
    )
    trained_at: str | None = Field(
        description=(
            "When the model was trained, ISO 8601 in UTC. Null until a trained "
            "model is selected."
        )
    )
    metrics: dict[str, float] | None = Field(
        description=(
            "The model's screening and backtest metrics. Null until a trained "
            "model is selected."
        )
    )
    note: str | None = Field(
        default=None, description="What's missing or stale, and what to do about it."
    )
    newest_cached: str | None = Field(
        description=(
            "Open time of the newest cached candle, ISO 8601 in UTC. Null when "
            "the cache is empty."
        )
    )
    newest_closed: str = Field(
        description="Open time of the most recently closed period, ISO 8601 in UTC."
    )
    is_current: bool = Field(
        description=(
            "True when the newest cached candle is the most recently closed "
            "period, so the next forecast uses current data."
        )
    )
```

Add the tool after `analyze_forecast`, before `return server`:

```python
    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_model_info() -> ModelInfoResult:
        """
        Describe the model behind the forecasts (name, kind, input window,
        training and metrics) and how fresh the cached data is, to judge how
        far to trust a forecast. Reads the cache only, and answers even when
        it is empty.
        """
        newest = read_candles("get_model_info", 1)
        newest_closed_ms = cache.newest_closed_ms()
        is_current = bool(newest) and newest[-1].timestamp == newest_closed_ms
        trained_through_ms = forecaster.trained_through_ms

        # The Forecaster Protocol has no training date or metrics yet: epic B
        # adds them with its trained model. trained_through_ms is the training
        # data's cutoff, not the date it was trained.
        notes = [
            f"{forecaster.name} is a {forecaster.kind} with no training date "
            "or screening and backtest metrics yet; they are filled in once a "
            "trained model is selected (epic B)."
        ]
        if not newest:
            notes.append(
                "The candle cache is empty: call get_candles to fill it, or have "
                "the operator run "
                f"`uv run fartt download --market {market} --interval {interval}`."
            )
        elif not is_current:
            notes.append(
                "The newest cached candle is older than the most recently "
                "closed period: call get_candles to update the cache before "
                "forecasting."
            )

        logger.info(
            f"get_model_info: {forecaster.name} ({forecaster.kind}) for "
            f"{market} {interval}, current={is_current}"
        )
        return ModelInfoResult(
            market=market,
            interval=interval,
            name=forecaster.name,
            kind=forecaster.kind,
            required_candles=forecaster.required_candles,
            trained_through=(
                None if trained_through_ms is None else _iso(trained_through_ms)
            ),
            trained_at=None,
            metrics=None,
            note=" ".join(notes),
            newest_cached=_iso(newest[-1].timestamp) if newest else None,
            newest_closed=_iso(newest_closed_ms),
            is_current=is_current,
        )
```

`src/fartt/server/__init__.py` becomes:

```python
from fartt.server.server import (
    AnalysisResult,
    CandlesResult,
    ForecastResult,
    ModelInfoResult,
    build_server,
)

__all__ = [
    "AnalysisResult",
    "CandlesResult",
    "ForecastResult",
    "ModelInfoResult",
    "build_server",
]
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_server.py -q 2>&1 | tail -1`
Expected: `29 passed`.

- [ ] **Step 4: Add `get_model_info` to the end-to-end check**

In `tests/e2e/test_server_stdio.py`:
- set `EXPECTED_TOOLS = ["get_candles", "get_forecast", "analyze_forecast", "get_model_info"]`;
- append to `test_trading_cycle_over_stdio`, after the `analyze_forecast` checks:

```python
        info = await client.call_tool("get_model_info", {})
        assert not info.is_error, info.content
        model = info.structured_content
        assert model["name"] == "repeat-last-return"
        assert model["kind"] == "naive baseline"
        assert model["required_candles"] == 2
        assert model["trained_at"] is None
        assert model["metrics"] is None
        assert model["newest_cached"] == content["rows"][-1][0]
        assert model["newest_closed"] == model["newest_cached"]
        assert model["is_current"] is True
```

Run: `task test:network 2>&1 | grep -E "passed|failed"`
Expected: `3 passed`.

- [ ] **Step 5: Update the docs**

- **`CLAUDE.md`:**
  - **Current-state server bullet:** "It has four tools: `get_candles`, `get_forecast`, `analyze_forecast` and `get_model_info`."
  - **"What doesn't exist yet":** delete the `get_model_info` (#73) bullet, since epic C is complete.
  - **The `serve` paragraph:** the agent calls all four tools.
  - **The `fartt/server/` bullet:**
    - "It has three tools." becomes "It has four tools.";
    - append that `get_model_info()` returns `name`, `kind`, `required_candles`, `trained_through` (the training data's cutoff, from `trained_through_ms`), `trained_at` and `metrics` (both `null` until epic B adds them to the Protocol), a `note`, `newest_cached`, `newest_closed` and `is_current`;
    - append that it answers even with an empty cache.
- **`README.md`:**
  - **Epic list:** item 2 (Forecast Tools) is marked done, like item 1.
  - **"What works today":** add "and what's behind it through `get_model_info`".
  - **"Today … exist" sentence:** name all four tools.
  - **Diagram:** if it has no `get_model_info` line, add `Agent->>Server: get_model_info` after `analyze_forecast`.
  - **Tree:** the server line names the four tools.

- [ ] **Step 6: Check, commit and push**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `198 passed, 3 deselected`.

```bash
git add src/fartt/server tests/test_server.py tests/e2e/test_server_stdio.py CLAUDE.md README.md
git commit -m "feat: add the get_model_info tool"
git push
```

When this merges, epic C is complete. The operator then runs the acceptance script on epic #45.
