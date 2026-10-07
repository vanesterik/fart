# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the candle cache to Claude Code as an MCP server with one tool, `get_candles`, started by `fartt serve` from the project's `.mcp.json` (story #52).

**Architecture:** `fartt.server.build_server(cache, market, interval)` returns an MCP SDK v2 `MCPServer` with a `get_candles` tool. The tool updates the `CandleCache` through the exchange layer, then returns the newest closed candles as a Pydantic model (structured output). An unreachable exchange degrades to cached data with `is_current: false` and a warning, or to a `ToolError` when nothing is cached. `fartt serve` validates its options, builds the exchange and cache, and runs the server over stdio. The exchange layer gains its own `ExchangeUnavailable` error and `supported_intervals()`, so neither the server nor the CLI imports ccxt. The cache gains a shared lock for readers.

**Tech Stack:** Python 3.11+, MCP Python SDK v2 (`mcp`), ccxt (via `fartt.exchange`), Pydantic, argparse, loguru, pytest with anyio, pyright (strict), uv, Task.

**Spec:** `docs/specs/2026-10-07-server-foundation-design.md` (section "3. MCP server and `get_candles`"), plus the scenarios issue #52 gained from the #50 and #51 reviews.

## Global Constraints

- `ccxt` is imported only in `src/fartt/exchange/ccxt_exchange.py`. The server and the CLI see `ExchangeUnavailable`, never ccxt's exceptions.
- The server serves **one** market and interval, fixed when it starts. `get_candles` takes only `count` (1 to 1000, default 50).
- Nothing writes to stdout while serving: stdout carries the MCP protocol. Logging goes through loguru's stderr and `logs/cli.log` sinks, which `cli.main()` sets up.
- `get_candles` is a plain `def`. MCP SDK v2 runs sync tools on a worker thread, so blocking ccxt calls don't stall the server (checked in the SDK docs, "What's new: sync functions run on a worker thread").
- Option errors exit with code 2 through `parser.error`, like the existing unknown-market error. An unreachable exchange exits `fartt download` with code 1 and a one-line message; `fartt serve` starts anyway and serves cached candles with a warning.
- `task check` passes before each commit; the pre-push gate runs it on push.
- Never reference "superpowers" in code, file paths or directory structure.

## Settled here

- **Exchange errors get a type the server can catch without ccxt.** The spec said ccxt's errors "pass through unchanged" for the cache and server to present. But catching `ccxt.NetworkError` in the server would import ccxt outside the exchange layer. Instead `CcxtExchange` re-raises ccxt's `NetworkError` and `ExchangeError` as `fartt.exchange.ExchangeUnavailable` (keeping the original as `__cause__`), and the server and CLI catch that.
- **Intervals are validated at startup against the exchange.** `Exchange.supported_intervals()` (ccxt's `timeframes`, so no network call) lets `fartt serve` and `fartt download` reject `3h` or `1x` with the supported list, instead of a traceback on the first fetch.
- **`fartt serve` starts even when the exchange is down.** The market check needs the exchange. If it's unreachable at startup, `serve` logs a warning and starts anyway, so the agent still gets cached candles flagged as stale; the first successful update proves the market. `fartt download` has nothing to serve without the exchange, so it exits with code 1.
- **"Current" has one definition.** `CandleCache.newest_closed_ms()` (`floor(now / interval) * interval - interval`) is used both by the update loop and by `get_candles`' `is_current`.
- **Readers take a shared lock.** `latest()` takes `LOCK_SH` on the same sidecar `.lock` file that `iter_update()` locks with `LOCK_EX`, so it never reads a half-written line from a concurrent `fartt download`.

## Review Focus

- **The exchange fails after some batches are written** (a long backfill on a flaky connection): expect the batches already appended to stay, and `get_candles` to return them with the partial `new_candles` count and a warning. Pinned in Task 3, `test_get_candles_keeps_batches_written_before_the_exchange_failed`.
- **The agent asks for more candles than are cached** (`count=1000` on a 3-candle cache): expect all 3, not an error. Pinned in Task 3, `test_get_candles_returns_what_is_cached_when_count_exceeds_it`.
- **`fartt serve` while the exchange is down at startup:** expect it to start and serve cached candles, not exit. Pinned in Task 4, `test_serve_starts_when_the_exchange_is_unreachable`.
- **A cache updated by another process while `get_candles` reads:** expect the read to wait for the write. Pinned in Task 2, `test_a_reader_waits_for_a_running_update`.
- **The exchange is up but hasn't published the newest closed candle yet** (a few seconds after a boundary): expect `is_current: false` with a warning saying so, not a silent stale answer. Pinned in Task 3, `test_get_candles_flags_a_missing_newest_candle`.

---

## Task 1: Exchange errors and supported intervals

**Files:**
- Modify: `src/fartt/exchange/exchange.py`, `src/fartt/exchange/ccxt_exchange.py`, `src/fartt/exchange/__init__.py`
- Modify: `tests/exchange/test_ccxt_exchange.py`
- Modify: `tests/fakes.py`

**Interfaces:**
- Produces:
  - `fartt.exchange.ExchangeUnavailable(Exception)`
  - `Exchange.supported_intervals(self) -> list[str]` (Protocol) and `CcxtExchange.supported_intervals()`
  - `CcxtExchange.has_market` and `.fetch_closed_candles` raise `ExchangeUnavailable` when ccxt raises `NetworkError` or `ExchangeError`
  - `tests/fakes.py::FakeExchange(candles, now_ms, markets=None, intervals=None, reachable=True, fail_after_fetches=None)` with `.supported_intervals()`; `reachable=False` makes `has_market` and `fetch_closed_candles` raise `ExchangeUnavailable`; `fail_after_fetches=n` lets `n` fetches succeed, then raises

- [ ] **Step 1: Write the failing exchange tests**

In `tests/exchange/test_ccxt_exchange.py`, change the imports and `StubClient` so it can raise, then append the new tests:

```python
import ccxt
```

(added below `from typing import Any`; tests may import ccxt, only `src/` may not outside the exchange layer)

```python
from fartt.exchange import Candle, CcxtExchange, ExchangeUnavailable
```

In `StubClient.__init__`, add a parameter `error: Exception | None = None` and store it as `self.error = error`. At the top of both `load_markets` and `fetch_ohlcv`, add:

```python
        if self.error is not None:
            raise self.error
```

Append:

```python
def test_supported_intervals_lists_the_exchange_timeframes() -> None:
    exchange = _exchange(StubClient(), now_ms=0)

    assert exchange.supported_intervals() == ["1m", "1h", "1d"]


def test_network_error_while_loading_markets_becomes_exchange_unavailable() -> None:
    client = StubClient(error=ccxt.NetworkError("connection reset"))
    exchange = _exchange(client, now_ms=0)

    with pytest.raises(ExchangeUnavailable, match="bitvavo.*connection reset") as raised:
        exchange.has_market("BTC/EUR")
    assert isinstance(raised.value.__cause__, ccxt.NetworkError)


def test_exchange_error_while_fetching_becomes_exchange_unavailable() -> None:
    client = StubClient()
    exchange = _exchange(client, now_ms=10 * HOUR_MS)
    exchange.has_market("BTC/EUR")  # markets load fine
    client.error = ccxt.ExchangeError("maintenance")

    with pytest.raises(ExchangeUnavailable, match="maintenance"):
        exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)
```

Run: `uv run pytest tests/exchange/test_ccxt_exchange.py -q 2>&1 | grep -E "Error|passed|failed" | head -3`
Expected: `ImportError: cannot import name 'ExchangeUnavailable'`.

- [ ] **Step 2: Add `ExchangeUnavailable` and `supported_intervals` to the interface**

In `src/fartt/exchange/exchange.py`, above `class Exchange`, add:

```python
class ExchangeUnavailable(Exception):
    """
    The exchange couldn't answer: a network failure, maintenance, a rate
    limit or another error on its side. Callers outside the exchange layer
    catch this instead of a library's own exceptions.

    """
```

In the `Exchange` Protocol, add below `has_market`:

```python
    def supported_intervals(self) -> list[str]:
        """Candle intervals the exchange offers, e.g. `["1m", "1h", "1d"]`."""
        ...
```

and add to the class docstring, after "Markets use ccxt's symbol format, e.g. `BTC/EUR`.":

```
    `has_market` and `fetch_closed_candles` raise `ExchangeUnavailable`
    when the exchange can't be reached or refuses the request.
```

In `src/fartt/exchange/__init__.py`:

```python
from fartt.exchange.candle import Candle
from fartt.exchange.ccxt_exchange import CcxtExchange
from fartt.exchange.exchange import Exchange, ExchangeUnavailable

__all__ = ["Candle", "CcxtExchange", "Exchange", "ExchangeUnavailable"]
```

- [ ] **Step 3: Translate ccxt's errors in `CcxtExchange`**

In `src/fartt/exchange/ccxt_exchange.py`, import `from fartt.exchange.exchange import ExchangeUnavailable`, then:

Add below `has_market`:

```python
    def supported_intervals(self) -> list[str]:
        return list(self._client.timeframes)
```

Wrap the `fetch_ohlcv` call in `fetch_closed_candles`:

```python
        try:
            rows = self._client.fetch_ohlcv(
                market, interval, since=since_ms, limit=limit
            )
        except (ccxt.NetworkError, ccxt.ExchangeError) as error:
            raise self._unavailable(error) from error
```

and the `load_markets` call in `_load_markets`:

```python
        if self._markets is None:
            try:
                self._markets = self._client.load_markets()
            except (ccxt.NetworkError, ccxt.ExchangeError) as error:
                raise self._unavailable(error) from error
        return self._markets
```

with the helper:

```python
    def _unavailable(self, error: Exception) -> ExchangeUnavailable:
        return ExchangeUnavailable(
            f"Exchange '{self.exchange_id}' is unavailable: {error}"
        )
```

- [ ] **Step 4: Extend the fake exchange**

Replace `FakeExchange.__init__`, `has_market` and the start of `fetch_closed_candles` in `tests/fakes.py` with:

```python
    def __init__(
        self,
        candles: list[Candle],
        now_ms: int,
        markets: set[str] | None = None,
        intervals: list[str] | None = None,
        reachable: bool = True,
        fail_after_fetches: int | None = None,
    ) -> None:
        self.candles = candles
        self.now_ms = now_ms
        self.markets = markets if markets is not None else {"BTC/EUR"}
        self.intervals = intervals if intervals is not None else ["1m", "1h"]
        self.reachable = reachable
        self.fail_after_fetches = fail_after_fetches
        self.fetch_calls: list[int] = []

    def has_market(self, market: str) -> bool:
        if not self.reachable:
            raise ExchangeUnavailable("Exchange 'fake' is unavailable: offline")
        return market in self.markets

    def supported_intervals(self) -> list[str]:
        return self.intervals

    def fetch_closed_candles(
        self, market: str, interval: str, since_ms: int, limit: int
    ) -> list[Candle]:
        if not self.has_market(market):
            raise ValueError(f"Market '{market}' not found on exchange 'fake'")
        if (
            self.fail_after_fetches is not None
            and len(self.fetch_calls) >= self.fail_after_fetches
        ):
            raise ExchangeUnavailable("Exchange 'fake' is unavailable: timed out")
        self.fetch_calls.append(since_ms)
```

(the rest of `fetch_closed_candles` is unchanged), and import `ExchangeUnavailable` alongside `Candle`: `from fartt.exchange import Candle, ExchangeUnavailable`.

- [ ] **Step 5: Run the tests and the full check**

Run: `uv run pytest tests/exchange/test_ccxt_exchange.py -q 2>&1 | tail -1`
Expected: `16 passed`.

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `112 passed, 1 deselected`.

Run: `grep -rn "import ccxt" src`
Expected: only `src/fartt/exchange/ccxt_exchange.py`.

- [ ] **Step 6: Commit**

```bash
git add src/fartt/exchange tests/exchange/test_ccxt_exchange.py tests/fakes.py
git commit -m "feat: report exchange outages and intervals without leaking ccxt"
```

---

## Task 2: Shared lock and the newest closed period

**Files:**
- Modify: `src/fartt/candle_cache.py`
- Modify: `tests/test_candle_cache.py`

**Interfaces:**
- Consumes: `FakeExchange(..., fail_after_fetches=...)`, `ExchangeUnavailable` (Task 1).
- Produces: `CandleCache.newest_closed_ms() -> int` (open time of the most recently closed period); `CandleCache.latest(count)` now waits for a running update.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_candle_cache.py` (it already imports `threading`), and add `from fartt.exchange import ExchangeUnavailable` to its imports:

```python
def test_newest_closed_ms_is_the_open_time_of_the_last_closed_period(
    tmp_path: Path,
) -> None:
    cache = _cache(FakeExchange([], now_ms=5 * HOUR_MS + HOUR_MS // 2), tmp_path)

    assert cache.newest_closed_ms() == 4 * HOUR_MS


def test_a_reader_waits_for_a_running_update(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    writer = _cache(exchange, tmp_path)
    reader = _cache(exchange, tmp_path)
    updates = writer.iter_update()
    assert next(updates) == 2  # the writer holds the lock mid-update

    read: list[list[Candle]] = []
    waiter = threading.Thread(target=lambda: read.append(reader.latest(10)))
    waiter.start()
    waiter.join(timeout=0.3)
    assert waiter.is_alive(), "latest() read while an update held the file"

    assert sum(updates) == 3
    waiter.join(timeout=5)
    assert read == [[candle(h) for h in range(5)]]


def test_update_keeps_batches_written_before_the_exchange_failed(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange(
        [candle(h) for h in range(5)], now_ms=5 * HOUR_MS, fail_after_fetches=1
    )
    cache = _cache(exchange, tmp_path)

    with pytest.raises(ExchangeUnavailable):
        cache.update()

    assert cache.latest(10) == [candle(0), candle(1)]
    exchange.fail_after_fetches = None
    assert cache.update() == 3
    assert _timestamps(cache.filepath) == [h * HOUR_MS for h in range(5)]
```

Run: `uv run pytest tests/test_candle_cache.py -q 2>&1 | tail -1`
Expected: `2 failed, 25 passed`: `newest_closed_ms` doesn't exist, and the reader doesn't wait. The partial-update test passes already, since batches are appended as they arrive. It pins that behaviour for #52's server.

- [ ] **Step 2: Implement `newest_closed_ms` and the shared lock**

In `src/fartt/candle_cache.py`:

Add a property and a public method to `CandleCache`, below `__init__`:

```python
    @property
    def _lock_path(self) -> Path:
        return self.filepath.with_name(f"{self.filepath.name}.lock")

    def newest_closed_ms(self) -> int:
        """Open time of the most recently closed period, by this cache's clock."""
        now_ms = self._clock()
        return (now_ms // self._interval_ms - 1) * self._interval_ms
```

In `iter_update`, replace the lock path line and use the property:

```python
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield from self._iter_update_locked()
```

In `_iter_update_locked`, replace

```python
        now_ms = self._clock()
        newest_closed_ms = (now_ms // self._interval_ms - 1) * self._interval_ms
```

with

```python
        newest_closed_ms = self.newest_closed_ms()
```

Replace `latest` with:

```python
    def latest(self, count: int) -> list[Candle]:
        """
        Return up to `count` of the newest cached candles, oldest first.

        Takes a shared lock on the same sidecar `.lock` file that updates
        lock exclusively, so it never reads a line another process is still
        writing. Readers don't block each other.

        """
        if not self.filepath.exists():
            return []
        with open(self._lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            with self.filepath.open(newline="", encoding="utf-8") as file:
                rows = deque(DictReader(file), maxlen=count)
        return [
            Candle(
                timestamp=int(row[TIMESTAMP]),
                open=float(row[OPEN]),
                high=float(row[HIGH]),
                low=float(row[LOW]),
                close=float(row[CLOSE]),
                volume=float(row[VOLUME]),
            )
            for row in rows
        ]
```

Update the class docstring's last paragraph to mention the shared lock: replace "is skipped." at its end with "is skipped. Updates hold an exclusive lock on a sidecar `.lock` file and `latest()` a shared one, so readers never see a half-written line."

- [ ] **Step 3: Run the tests and the full check**

Run: `uv run pytest tests/test_candle_cache.py -q 2>&1 | tail -1`
Expected: `27 passed`.

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `115 passed, 1 deselected`.

- [ ] **Step 4: Commit**

```bash
git add src/fartt/candle_cache.py tests/test_candle_cache.py
git commit -m "feat: let cache readers wait for writers and expose the newest closed period"
```

---

## Task 3: The MCP server with `get_candles`

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (`uv add mcp`)
- Create: `src/fartt/server/__init__.py`, `src/fartt/server/server.py`
- Create: `tests/test_server.py`

**Interfaces:**
- Consumes: `CandleCache(...)`, `.iter_update()`, `.latest(count)`, `.newest_closed_ms()` (Task 2); `ExchangeUnavailable` (Task 1); `FakeExchange(..., reachable=..., fail_after_fetches=...)`, `candle`, `HOUR_MS` (Task 1).
- Produces: `fartt.server.build_server(cache: CandleCache, market: str, interval: str) -> MCPServer`; the tool `get_candles(count: int = 50) -> CandlesResult`.

- [ ] **Step 1: Add the SDK**

Run: `uv add "mcp>=2.3"`
Expected: `mcp` 2.3 or newer in `pyproject.toml` and `uv.lock`.

- [ ] **Step 2: Write the failing server tests**

Create `tests/test_server.py`:

```python
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.candle_cache import CandleCache
from fartt.server import build_server
from tests.fakes import HOUR_MS, FakeExchange, candle


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _cache(exchange: FakeExchange, tmp_path: Path, batch_limit: int = 2) -> CandleCache:
    return CandleCache(
        exchange=exchange,
        assets_dir=tmp_path,
        market="BTC/EUR",
        interval="1h",
        history_start_ms=0,
        batch_limit=batch_limit,
        clock=lambda: exchange.now_ms,
    )


async def _get_candles(cache: CandleCache, **arguments: Any) -> Any:
    async with Client(build_server(cache, "BTC/EUR", "1h")) as client:
        return await client.call_tool("get_candles", arguments)


@pytest.mark.anyio
async def test_get_candles_returns_latest_closed_candles(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    result = await _get_candles(_cache(exchange, tmp_path), count=2)

    assert not result.is_error
    assert result.structured_content == {
        "market": "BTC/EUR",
        "interval": "1h",
        "candles": [
            {"time": "1970-01-01T01:00:00+00:00", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0, "volume": 2.5},
            {"time": "1970-01-01T02:00:00+00:00", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0, "volume": 2.5},
        ],
        "new_candles": 3,
        "is_current": True,
        "warning": None,
    }


@pytest.mark.anyio
async def test_get_candles_brings_the_cache_up_to_date(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)

    await _get_candles(cache)

    assert cache.latest(10) == [candle(h) for h in range(3)]


@pytest.mark.anyio
async def test_get_candles_returns_what_is_cached_when_count_exceeds_it(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    result = await _get_candles(_cache(exchange, tmp_path), count=1000)

    assert len(result.structured_content["candles"]) == 3


@pytest.mark.anyio
async def test_get_candles_rejects_count_out_of_range(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)

    for count in (0, 1001):
        result = await _get_candles(cache, count=count)
        assert result.is_error
        assert "count" in result.content[0].text


@pytest.mark.anyio
async def test_get_candles_serves_stale_cache_when_exchange_is_unreachable(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.reachable = False
    exchange.now_ms = 5 * HOUR_MS  # two more periods closed meanwhile

    result = await _get_candles(cache, count=2)

    content = result.structured_content
    assert not result.is_error
    assert [c["time"] for c in content["candles"]] == [
        "1970-01-01T01:00:00+00:00",
        "1970-01-01T02:00:00+00:00",
    ]
    assert content["new_candles"] == 0
    assert content["is_current"] is False
    assert "unavailable" in content["warning"]
    assert "hold" in content["warning"]


@pytest.mark.anyio
async def test_get_candles_keeps_batches_written_before_the_exchange_failed(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange(
        [candle(h) for h in range(5)], now_ms=5 * HOUR_MS, fail_after_fetches=1
    )

    result = await _get_candles(_cache(exchange, tmp_path), count=10)

    content = result.structured_content
    assert len(content["candles"]) == 2
    assert content["new_candles"] == 2
    assert content["is_current"] is False
    assert "unavailable" in content["warning"]


@pytest.mark.anyio
async def test_get_candles_flags_a_missing_newest_candle(tmp_path: Path) -> None:
    # Reachable, but the exchange hasn't published the 2h candle yet.
    exchange = FakeExchange([candle(0), candle(1)], now_ms=3 * HOUR_MS)

    result = await _get_candles(_cache(exchange, tmp_path), count=2)

    content = result.structured_content
    assert content["is_current"] is False
    assert "not published" in content["warning"]


@pytest.mark.anyio
async def test_get_candles_errors_when_nothing_is_cached_and_exchange_is_down(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange([], now_ms=3 * HOUR_MS, reachable=False)

    result = await _get_candles(_cache(exchange, tmp_path))

    assert result.is_error
    assert "No BTC/EUR 1h candles are cached" in result.content[0].text
    assert "Retry next cycle" in result.content[0].text


@pytest.mark.anyio
async def test_server_offers_get_candles_as_a_read_only_tool(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)

    async with Client(build_server(cache, "BTC/EUR", "1h")) as client:
        tools = (await client.list_tools()).tools

    assert [tool.name for tool in tools] == ["get_candles"]
    assert tools[0].annotations is not None
    assert tools[0].annotations.read_only_hint is True
    assert tools[0].input_schema["properties"]["count"]["maximum"] == 1000
```

Run: `uv run pytest tests/test_server.py -q 2>&1 | grep -E "Error" | head -2`
Expected: `ModuleNotFoundError: No module named 'fartt.server'`.

- [ ] **Step 3: Implement the server**

Create `src/fartt/server/__init__.py`:

```python
from fartt.server.server import CandlesResult, build_server

__all__ = ["CandlesResult", "build_server"]
```

Create `src/fartt/server/server.py`:

```python
from datetime import UTC, datetime
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fartt.candle_cache import CandleCache
from fartt.exchange import Candle, ExchangeUnavailable

MAX_CANDLES = 1000

INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Start every trading cycle with "
    "get_candles. If is_current is false the data is stale: read the warning "
    "and treat the cycle as a hold."
)


class CandleOut(BaseModel):
    time: str = Field(description="Candle open time, ISO 8601 in UTC.")
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesResult(BaseModel):
    market: str
    interval: str
    candles: list[CandleOut] = Field(
        description="Closed candles, oldest first; the last is the newest."
    )
    new_candles: int = Field(
        description="Candles fetched from the exchange by this call."
    )
    is_current: bool = Field(
        description=(
            "True when the last candle is the most recently closed period. "
            "False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="Why the data may be stale, if it may be."
    )


def _candle_out(candle: Candle) -> CandleOut:
    return CandleOut(
        time=datetime.fromtimestamp(candle.timestamp / 1000, UTC).isoformat(),
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


def build_server(cache: CandleCache, market: str, interval: str) -> MCPServer:
    """
    Build the MCP server for one market and interval, backed by `cache`.

    A factory rather than a module-level server, so tests can pass a cache
    over an in-memory exchange and connect with the SDK's in-memory client.

    """
    server = MCPServer("fartt", instructions=INSTRUCTIONS)

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True)
    )
    def get_candles(
        count: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_CANDLES,
                description="How many of the latest closed candles to return.",
            ),
        ] = 50,
    ) -> CandlesResult:
        """
        Return the latest closed candles for the server's market and
        interval, oldest first, after bringing the local cache up to date
        from the exchange.
        """
        new_candles = 0
        warning: str | None = None
        try:
            for batch in cache.iter_update():
                new_candles += batch
        except ExchangeUnavailable as error:
            warning = (
                f"{error}. These are the cached candles, which may be stale: "
                "treat this cycle as a hold, or retry next cycle."
            )

        candles = cache.latest(count)
        if not candles:
            raise ToolError(
                f"No {market} {interval} candles are cached yet"
                + (" and the exchange is unreachable" if warning else "")
                + ". Retry next cycle."
            )

        is_current = candles[-1].timestamp == cache.newest_closed_ms()
        if not is_current and warning is None:
            warning = (
                "The most recently closed candle is not published by the "
                "exchange yet; the last candle here is the one before it. "
                "Retry shortly or treat this cycle as a hold."
            )

        return CandlesResult(
            market=market,
            interval=interval,
            candles=[_candle_out(c) for c in candles],
            new_candles=new_candles,
            is_current=is_current,
            warning=warning,
        )

    return server
```

- [ ] **Step 4: Run the server tests**

Run: `uv run pytest tests/test_server.py -q 2>&1 | tail -1`
Expected: `9 passed`.

- [ ] **Step 5: Run the full check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `124 passed, 1 deselected`.

Run: `grep -rn "import ccxt" src`
Expected: only `src/fartt/exchange/ccxt_exchange.py`.

```bash
git add pyproject.toml uv.lock src/fartt/server tests/test_server.py
git commit -m "feat: add MCP server with get_candles"
```

---

## Task 4: `fartt serve`, option validation and registration

**Files:**
- Modify: `src/fartt/cli.py`
- Modify: `tests/test_cli.py`
- Create: `.mcp.json`
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `build_server` (Task 3); `CcxtExchange.supported_intervals()`, `ExchangeUnavailable` (Task 1); `FakeExchange(..., intervals=..., reachable=...)` (Task 1).
- Produces: the CLI `fartt serve [--exchange bitvavo] [--market BTC/EUR] [--interval 1h] [--assets-dir assets]`; `fartt download` gains the same option validation.

- [ ] **Step 1: Write the failing CLI tests**

In `tests/test_cli.py`, replace the `fake_exchange` fixture so an unknown exchange id fails like the real one, and add a fixture that captures the server instead of running it:

```python
@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeExchange:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    def make_exchange(exchange_id: str) -> FakeExchange:
        if exchange_id != "bitvavo":
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")
        return exchange

    monkeypatch.setattr(cli, "CcxtExchange", make_exchange)
    monkeypatch.setattr(cli, "DEFAULT_HISTORY_START_MS", 0)
    monkeypatch.chdir(tmp_path)  # main() writes logs/cli.log relative to the cwd
    return exchange


class FakeServer:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    built: dict[str, Any] = {}

    def fake_build_server(cache: Any, market: str, interval: str) -> FakeServer:
        built.update(cache=cache, market=market, interval=interval, server=FakeServer())
        return built["server"]

    monkeypatch.setattr(cli, "build_server", fake_build_server)
    return built
```

(add `from typing import Any` to the imports). Then append:

```python
@pytest.mark.parametrize("interval", ["3h", "1x"])
def test_download_rejects_an_interval_the_exchange_does_not_offer(
    fake_exchange: FakeExchange,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    interval: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--interval", interval])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert f"interval '{interval}'" in error
    assert "1m, 1h" in error


def test_download_rejects_an_unknown_exchange(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--exchange", "bitvavoo"])

    assert exit_info.value.code == 2
    assert "bitvavoo" in capsys.readouterr().err


def test_download_exits_cleanly_when_the_exchange_is_unreachable(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_exchange.reachable = False

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--interval", "1h"])

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "unavailable" in error
    assert "Traceback" not in error


def test_serve_runs_the_server_for_the_configured_market(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    cli.main(["serve", "--assets-dir", str(tmp_path), "--market", "BTC/EUR", "--interval", "1h"])

    assert served["market"] == "BTC/EUR"
    assert served["interval"] == "1h"
    assert served["cache"].filepath == tmp_path / "BTC-EUR-1h.csv"
    assert served["server"].ran


def test_serve_starts_when_the_exchange_is_unreachable(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    fake_exchange.reachable = False

    cli.main(["serve", "--assets-dir", str(tmp_path), "--interval", "1h"])

    assert served["server"].ran


def test_serve_rejects_an_unknown_market(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--assets-dir", str(tmp_path), "--market", "BTC-EUR", "--interval", "1h"])

    assert exit_info.value.code == 2
    assert "BTC/EUR" in capsys.readouterr().err
    assert not served
```

Run: `uv run pytest tests/test_cli.py -q 2>&1 | tail -1`
Expected: failures and errors: `fartt.cli` has no `build_server`, no `serve` command, and interval/exchange errors still surface as tracebacks.

- [ ] **Step 2: Rewrite the CLI's command handling**

In `src/fartt/cli.py`:

Imports: add `from fartt.exchange import CcxtExchange, ExchangeUnavailable` (replacing the plain `CcxtExchange` import) and `from fartt.server import build_server`.

In `build_parser()`, move the four options into a helper and add `serve`:

```python
def _add_cache_options(command: argparse.ArgumentParser, interval: str) -> None:
    command.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="folder the candle cache lives in (default: %(default)s)",
    )
    command.add_argument(
        "--exchange", default="bitvavo", help="ccxt exchange id (default: %(default)s)"
    )
    command.add_argument(
        "--market",
        default="BTC/EUR",
        help="market in ccxt's format, e.g. BTC/EUR (default: %(default)s)",
    )
    command.add_argument(
        "--interval",
        default=interval,
        help="candle interval the exchange offers, e.g. 1m, 30m, 1h, 4h, 1d "
        "(default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fartt", description="Fartt: Financial Analysis Real Time Trading."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download",
        help="download closed candles into the local cache",
        description="Download closed candles into the local cache, resuming where it left off.",
    )
    _add_cache_options(download, interval="1d")

    serve = commands.add_parser(
        "serve",
        help="run the MCP server over stdio (started by Claude Code from .mcp.json)",
        description="Run the MCP server for one market and interval over stdio.",
    )
    _add_cache_options(serve, interval="1h")
    return parser
```

In `main()`, dispatch both commands:

```python
    if args.command == "download":
        _download(parser, args.assets_dir, args.exchange, args.market, args.interval)
    elif args.command == "serve":
        _serve(parser, args.assets_dir, args.exchange, args.market, args.interval)
```

Replace `_download` and add `_open_cache` and `_serve`:

```python
def _open_cache(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
    require_exchange: bool,
) -> CandleCache:
    # Validate the options before anything touches the cache, so a typo
    # exits with a one-line message rather than a traceback.
    try:
        client = CcxtExchange(exchange_id=exchange)
    except ValueError:
        parser.error(f"unknown exchange '{exchange}'; use a ccxt exchange id, e.g. 'bitvavo'")

    supported = client.supported_intervals()
    if interval not in supported:
        parser.error(
            f"interval '{interval}' is not offered by exchange '{exchange}' "
            f"(supported: {', '.join(supported)})"
        )

    try:
        known = client.has_market(market)
    except ExchangeUnavailable as error:
        if require_exchange:
            parser.exit(1, f"fartt: error: {error}\n")
        logger.warning(f"{error}; starting anyway without checking market '{market}'")
        known = True
    if not known:
        parser.error(
            f"market '{market}' not found on exchange '{exchange}'. "
            "Markets use ccxt's format, e.g. 'BTC/EUR' rather than 'BTC-EUR'."
        )

    return CandleCache(
        exchange=client,
        assets_dir=assets_dir,
        market=market,
        interval=interval,
        history_start_ms=DEFAULT_HISTORY_START_MS,
    )


def _download(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
) -> None:
    cache = _open_cache(parser, assets_dir, exchange, market, interval, require_exchange=True)
    configuration = {
        "exchange": exchange,
        "market": market,
        "interval": interval,
        "filepath": str(cache.filepath),
    }
    logger.info(f"\n\nFartt Downloader\n\n{tabulate(configuration.items())}\n")

    try:
        with tqdm(desc="Downloading", unit=" candles") as progress:
            for count in cache.iter_update():
                progress.update(count)
    except ExchangeUnavailable as error:
        parser.exit(1, f"fartt: error: {error}; the candles fetched so far are kept\n")


def _serve(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
) -> None:
    cache = _open_cache(parser, assets_dir, exchange, market, interval, require_exchange=False)
    logger.info(f"Serving {market} {interval} candles from {cache.filepath} over stdio")
    build_server(cache, market, interval).run()
```

- [ ] **Step 3: Run the CLI tests**

Run: `uv run pytest tests/test_cli.py -q 2>&1 | tail -1`
Expected: `10 passed` (3 existing, 7 new: two interval cases, unknown exchange, unreachable download, three serve tests).

Run: `uv run fartt serve --help`
Expected: the four options, with `--interval` defaulting to `1h`.

- [ ] **Step 4: Register the server for Claude Code**

Create `.mcp.json`:

```json
{
  "mcpServers": {
    "fartt": {
      "command": "uv",
      "args": ["run", "fartt", "serve", "--market", "BTC/EUR", "--interval", "1h"]
    }
  }
}
```

- [ ] **Step 5: Check it end to end over stdio against Bitvavo**

Run (a throwaway client, not committed):

```bash
uv run python - <<'EOF'
import anyio
from mcp import Client, StdioServerParameters

async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "fartt", "serve", "--assets-dir", "/tmp/fartt-serve-check",
              "--market", "BTC/EUR", "--interval", "1d"],
    )
    async with Client(params) as client:
        print([t.name for t in (await client.list_tools()).tools])
        result = await client.call_tool("get_candles", {"count": 3})
        content = result.structured_content
        print(result.is_error, content["new_candles"], content["is_current"], content["warning"])
        print(content["candles"][-1])

anyio.run(main)
EOF
rm -rf /tmp/fartt-serve-check
```

Expected: `['get_candles']`, then `False <about 2770> True None` (a fresh daily cache filled through the tool), then yesterday's closed BTC/EUR daily candle with an ISO time ending `T00:00:00+00:00`. Nothing else on stdout: the server's logs go to stderr.

If `StdioServerParameters` isn't importable from `mcp`, use `from mcp.client.stdio import StdioServerParameters` and record a ruling.

- [ ] **Step 6: Update the documentation**

In `CLAUDE.md`:
- In the Commands block, add after the `uv run fartt download` line: `uv run fartt serve                # run the MCP server over stdio (Claude Code starts it from .mcp.json)`.
- After the `fartt download` paragraph, add: "`fartt serve` takes the same four options (`--interval` defaults to `1h`) and runs the MCP server (`fartt/server/server.py`) over stdio. Claude Code starts it from the project's `.mcp.json`; the agent calls its `get_candles` tool. Both commands reject an unknown exchange, an interval the exchange doesn't offer or an unknown market with exit code 2; an unreachable exchange stops `download` with exit code 1, while `serve` starts anyway and serves cached candles flagged as stale."
- In Architecture, add a bullet: "**`fartt/server/`** — `build_server(cache, market, interval)` returns the MCP SDK v2 `MCPServer` with one tool, `get_candles(count=1..1000)`: it updates the `CandleCache` and returns the newest closed candles as structured output (`market`, `interval`, `candles`, `new_candles`, `is_current`, `warning`). An unreachable exchange (`ExchangeUnavailable` from the exchange layer) degrades to cached candles with `is_current: false` and a warning, or a `ToolError` when nothing is cached."

In `README.md`'s Usage section, after the download paragraph, add:

```markdown
The MCP server exposes the cache to Claude Code. It's registered in `.mcp.json`, so a Claude Code session in this directory starts it and the agent can call `get_candles`. Run `fartt download` with the same market and interval first, so the first call doesn't have to fill years of history:

```bash
uv run fartt download --market BTC/EUR --interval 1h
```
```

- [ ] **Step 7: Run the full check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `131 passed, 1 deselected`.

```bash
git add src/fartt/cli.py tests/test_cli.py .mcp.json CLAUDE.md README.md
git commit -m "feat: add fartt serve and register the MCP server for Claude Code"
```

- [ ] **Step 8: Push (runs the pre-push gate)**

Run: `git push`
Expected: lefthook's `task check` passes, then the push goes through.

- [ ] **Manual acceptance (operator):** open a new Claude Code session in the project, approve the `fartt` server when asked, check `/mcp` lists it as connected, and ask the agent for the latest candles.
