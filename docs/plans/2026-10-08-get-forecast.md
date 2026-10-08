# get_forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `get_forecast` tool: the forecaster's expected return for the next candle of the server's market and interval, read from the cache (story #71).

**Architecture:** `build_server` gains a `forecaster: Forecaster` argument (the analysis settings follow in #72); `fartt serve` passes `RepeatLastReturn()`. The new tool reads `forecaster.required_candles` candles from the cache, calls `fartt.forecast.forecast()`, and returns a `ForecastResult`. It follows `get_candles`' conventions (ISO 8601 UTC times, `is_current` plus `warning`, and a `ToolError` saying what to do), but it is read-only and never contacts the exchange. The end-to-end check gains the tool.

**Tech Stack:** MCP Python SDK v2, Pydantic, pytest with anyio, pyright (strict).

**Spec:** `docs/specs/2026-10-08-forecast-tools-design.md` (section "3. The tools", and Decisions on units)

## Global Constraints

- `expected_return` is a fraction (`0.004` = +0.4%), and its field description says so.
- `applies_to` and `based_on` are candle **open** times in ISO 8601 UTC, like `get_candles`' `time` column. Their field descriptions say so.
- Read-only and closed-world: `ToolAnnotations(read_only_hint=True, open_world_hint=False)`. The tool reads the cache and never updates it or calls the exchange; `get_candles` starts each cycle.
- `is_current` means the forecast is based on the most recently closed period (`based_on_ms == cache.newest_closed_ms()`). When it's false, the `warning` says to call `get_candles` first, and to treat the cycle as a hold if the exchange is unreachable.
- Too little history is a `ToolError` that names the candles needed and cached, and the exact command for the server's own cache: `uv run fartt download --market <market> --interval <interval>`.
- `build_server(cache, market, interval, forecaster)`: the forecaster is required (no default), so the wiring is explicit.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **The agent calls `get_forecast` before `get_candles` in a new cycle** (a new period has closed, but the cache hasn't caught up): expect a forecast with `is_current: false` and a warning saying to call `get_candles` first, not a silently stale "current" forecast. Pinned by `test_get_forecast_flags_a_stale_cache`.
- **An empty cache** (a fresh install, before any download): expect a `ToolError` with the exact download command for this server's market and interval, not a traceback. Pinned by `test_get_forecast_explains_too_little_history`.
- **The exchange is down:** expect `get_forecast` to work anyway, because it never calls the exchange. Pinned by `test_get_forecast_never_calls_the_exchange` (the fake exchange is unreachable during the call).
- **A falling market:** expect a negative `expected_return`, and the sign preserved through the structured output. Pinned by `test_get_forecast_keeps_the_sign_of_a_fall`.
- **A cache file that can't be read:** expect a `ToolError` naming the file, the same convention as `get_candles`. Not separately tested: `get_candles` pins that convention, and the code path is the same `OSError` → `ToolError` translation.

---

## Task 1: The `get_forecast` tool

**Files:**
- Modify: `src/fartt/server/server.py`, `src/fartt/cli.py`
- Modify: `tests/test_server.py`, `tests/test_cli.py`, `tests/e2e/test_server_stdio.py`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `fartt.forecast.Forecaster`, `forecast()`, `NotEnoughCandles` (`.needed`, `.got`), `RepeatLastReturn` (#70); `CandleCache.latest()`, `.newest_closed_ms()`, `.filepath`; `fartt.candle_cache.interval_to_ms`.
- Produces: `build_server(cache: CandleCache, market: str, interval: str, forecaster: Forecaster) -> MCPServer`; the tool `get_forecast() -> ForecastResult` (`market`, `interval`, `model`, `expected_return`, `applies_to`, `based_on`, `is_current`, `warning`).

- [ ] **Step 1: Update the server tests' helpers to the new signature, and add the failing tests**

In `tests/test_server.py`, add `from fartt.forecast import RepeatLastReturn`, then add this helper below `_cache` and use it in place of every `build_server(cache, "BTC/EUR", "1h")` call (two places: `_get_candles` and the tool-list test):

```python
def _server(cache: CandleCache) -> Any:
    return build_server(cache, "BTC/EUR", "1h", RepeatLastReturn())


async def _get_forecast(cache: CandleCache) -> Any:
    async with Client(_server(cache)) as client:
        return await client.call_tool("get_forecast", {})
```

Change the tool-list test's assertion to `assert [tool.name for tool in tools] == ["get_candles", "get_forecast"]`, and append to that test:

```python
    forecast_tool = tools[1]
    assert forecast_tool.annotations is not None
    assert forecast_tool.annotations.read_only_hint is True
    assert forecast_tool.annotations.open_world_hint is False
    assert forecast_tool.input_schema.get("properties", {}) == {}
    assert "fraction" in forecast_tool.output_schema["properties"]["expected_return"]["description"]
```

Append the new tests:

```python
@pytest.mark.anyio
async def test_get_forecast_returns_the_next_candles_expected_return(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange(
        [candle(0, close=100.0), candle(1, close=100.0), candle(2, close=101.0)],
        now_ms=3 * HOUR_MS,
    )
    cache = _cache(exchange, tmp_path)
    cache.update()

    result = await _get_forecast(cache)

    assert not result.is_error, result.content
    content = result.structured_content
    assert content == {
        "market": "BTC/EUR",
        "interval": "1h",
        "model": "repeat-last-return",
        "expected_return": pytest.approx(0.01),
        "applies_to": "1970-01-01T03:00:00+00:00",
        "based_on": "1970-01-01T02:00:00+00:00",
        "is_current": True,
        "warning": None,
    }


@pytest.mark.anyio
async def test_get_forecast_keeps_the_sign_of_a_fall(tmp_path: Path) -> None:
    exchange = FakeExchange(
        [candle(0, close=100.0), candle(1, close=98.0)], now_ms=2 * HOUR_MS
    )
    cache = _cache(exchange, tmp_path)
    cache.update()

    content = (await _get_forecast(cache)).structured_content

    assert content["expected_return"] == pytest.approx(-0.02)


@pytest.mark.anyio
async def test_get_forecast_flags_a_stale_cache(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.now_ms = 5 * HOUR_MS  # two more periods closed; cache not updated

    content = (await _get_forecast(cache)).structured_content

    assert content["is_current"] is False
    assert content["based_on"] == "1970-01-01T02:00:00+00:00"
    assert "get_candles" in content["warning"]


@pytest.mark.anyio
async def test_get_forecast_explains_too_little_history(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=3 * HOUR_MS), tmp_path)

    result = await _get_forecast(cache)

    assert result.is_error
    text = result.content[0].text
    assert "needs 2" in text
    assert "holds 0" in text
    assert "uv run fartt download --market BTC/EUR --interval 1h" in text


@pytest.mark.anyio
async def test_get_forecast_never_calls_the_exchange(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.fetch_calls.clear()
    exchange.reachable = False  # it would raise if called

    result = await _get_forecast(cache)

    assert not result.is_error, result.content
    assert exchange.fetch_calls == []
```

In `tests/test_cli.py`, change the `served` fixture's fake to accept and record the forecaster, and check it in `test_serve_runs_the_server_for_the_configured_market`:

```python
    def fake_build_server(
        cache: Any, market: str, interval: str, forecaster: Any
    ) -> FakeServer:
        built.update(
            cache=cache,
            market=market,
            interval=interval,
            forecaster=forecaster,
            server=FakeServer(),
        )
        return built["server"]
```

and add `assert served["forecaster"] == RepeatLastReturn()` to that test (importing `RepeatLastReturn` from `fartt.forecast`).

Run: `uv run pytest tests/test_server.py tests/test_cli.py -q 2>&1 | tail -1`
Expected: failures and errors: `build_server` doesn't take a forecaster, and `get_forecast` doesn't exist.

- [ ] **Step 2: Add the tool**

In `src/fartt/server/server.py`:

Imports: `from fartt.candle_cache import CandleCache, interval_to_ms` and `from fartt.forecast import Forecaster, NotEnoughCandles, forecast`.

Replace `INSTRUCTIONS` with:

```python
INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Every trading cycle starts with "
    "get_candles, which updates the cache from the exchange, then "
    "get_forecast for the next candle's expected return. If is_current is "
    "false the data is stale: read the warning and treat the cycle as a hold."
)
```

Add a helper below `Row` and use it in `_row` (replacing its inline `datetime.fromtimestamp(...)`):

```python
def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat()
```

Add the result model below `CandlesResult`:

```python
class ForecastResult(BaseModel):
    market: str
    interval: str
    model: str = Field(description="Name of the model that made the forecast.")
    expected_return: float = Field(
        description=(
            "Expected return of the next candle, as a fraction: 0.004 means "
            "+0.4%. The sign is the direction."
        )
    )
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
```

Change the signature to `def build_server(cache: CandleCache, market: str, interval: str, forecaster: Forecaster) -> MCPServer:`, add `interval_ms = interval_to_ms(interval)` at its top, and add the tool after `get_candles`, before `return server`:

```python
    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_forecast() -> ForecastResult:
        """
        Forecast the return of the next candle of the server's market and
        interval from the cached candles. Call get_candles first: this tool
        reads the cache and never contacts the exchange.
        """
        try:
            candles = cache.latest(forecaster.required_candles)
        except OSError as error:
            logger.error(f"get_forecast: {error}")
            raise ToolError(
                f"Couldn't read the candle cache at {cache.filepath}: {error}. "
                "Retrying won't help until the operator fixes the file or its "
                "folder."
            ) from error

        try:
            result = forecast(forecaster, candles, interval_ms)
        except NotEnoughCandles as error:
            raise ToolError(
                f"{forecaster.name} needs {error.needed} closed {market} "
                f"{interval} candles and the cache holds {error.got}. Call "
                "get_candles to update it, or have the operator fill it with "
                f"`uv run fartt download --market {market} --interval {interval}`; "
                "otherwise retry next cycle."
            ) from error

        is_current = result.based_on_ms == cache.newest_closed_ms()
        warning = None
        if not is_current:
            warning = (
                "The newest cached candle is older than the most recently "
                "closed period, so this forecast may be stale. Call get_candles "
                "first to update the cache; if it reports the exchange "
                "unreachable, treat this cycle as a hold."
            )

        logger.info(
            f"get_forecast: {result.model} expects {result.expected_return:+.4%} "
            f"for {market} {interval}, current={is_current}"
        )
        return ForecastResult(
            market=market,
            interval=interval,
            model=result.model,
            expected_return=result.expected_return,
            applies_to=_iso(result.applies_to_ms),
            based_on=_iso(result.based_on_ms),
            is_current=is_current,
            warning=warning,
        )
```

Export it: `src/fartt/server/__init__.py` becomes

```python
from fartt.server.server import CandlesResult, ForecastResult, build_server

__all__ = ["CandlesResult", "ForecastResult", "build_server"]
```

In `src/fartt/cli.py::_serve`, import `from fartt.forecast import RepeatLastReturn` and change the last line to `build_server(cache, market, interval, RepeatLastReturn()).run()`.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_server.py tests/test_cli.py -q 2>&1 | tail -1`
Expected: `29 passed` (14 server tests: 9 + 5 new; 15 CLI tests).

- [ ] **Step 4: Add `get_forecast` to the end-to-end check**

In `tests/e2e/test_server_stdio.py`, set `EXPECTED_TOOLS = ["get_candles", "get_forecast"]`, and append to `test_trading_cycle_over_stdio`, inside the `async with` block, after the `get_candles` checks:

```python
        forecast = await client.call_tool("get_forecast", {})
        assert not forecast.is_error, forecast.content
        prediction = forecast.structured_content
        assert prediction["market"] == "BTC/EUR"
        assert prediction["model"] == "repeat-last-return"
        assert isinstance(prediction["expected_return"], float)
        assert prediction["based_on"] == content["rows"][-1][0]
        assert prediction["applies_to"] > prediction["based_on"]
        assert prediction["is_current"] is True
```

Run: `task test:network 2>&1 | grep -E "passed|failed"`
Expected: `3 passed`.

- [ ] **Step 5: Update the docs**

- `CLAUDE.md`, in the Architecture `fartt/server/` bullet:
  - say the server has two tools so far: `get_candles` as described, and `get_forecast()`. The latter reads `forecaster.required_candles` candles from the cache (never the exchange), and returns `model`, `expected_return` (a fraction), `applies_to` and `based_on` (candle open times), `is_current` and `warning`. Too little history is a `ToolError` with the exact `fartt download` command;
  - add that `build_server(cache, market, interval, forecaster)` takes the forecaster, and that `fartt serve` passes `RepeatLastReturn()`.

  In "What doesn't exist yet", the forecast-tools bullet becomes: "`analyze_forecast` and `get_model_info` (#72, #73). `get_forecast` exists, backed by the `RepeatLastReturn` baseline from `fartt/forecast/`."
- `README.md`, in "How a trading cycle works":
  - "Today only `get_candles` exists" becomes "Today `get_candles` and `get_forecast` exist";
  - in the diagram, the line `Agent->>Server: get_forecast, analyze_forecast (planned)` becomes two lines, `Agent->>Server: get_forecast` and `Agent->>Server: analyze_forecast (planned)`.

- [ ] **Step 6: Check, commit and push**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `154 passed, 3 deselected`.

```bash
git add src/fartt/server src/fartt/cli.py tests/test_server.py tests/test_cli.py tests/e2e/test_server_stdio.py CLAUDE.md README.md
git commit -m "feat: add the get_forecast tool"
git push
```
