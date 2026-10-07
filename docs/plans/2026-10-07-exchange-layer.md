# Exchange Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch closed candles through ccxt behind an internal `Exchange` interface (story #50), with a pre-push quality gate in place before any code lands.

**Architecture:** A new `fart.exchange` package holds a `Candle` NamedTuple, an `Exchange` Protocol (`has_market`, `fetch_closed_candles`) and `CcxtExchange`, the only module that imports ccxt. `CcxtExchange` builds a ccxt client from an exchange ID (default `bitvavo`), loads markets once, drops the still-forming candle by comparing each candle's end with an injectable clock, and returns candles sorted by timestamp. Nothing uses the layer yet; the candle cache (story #51) is its first caller.

**Tech Stack:** Python 3.11+, ccxt 4.5, pytest, pyright (strict), ruff, lefthook, uv.

**Spec:** `docs/specs/2026-10-07-server-foundation-design.md` (sections "Exchange layer" and "Quality gate")

## Global Constraints

- Market symbols use ccxt's format, `BTC/EUR`. File names keep `BTC-EUR`, but that conversion belongs to story #51, not this plan.
- `ccxt` is imported only inside `src/fart/exchange/ccxt_exchange.py` (spec, story #50 acceptance criterion).
- `Exchange` is a `typing.Protocol`, like `fart.model.builder.ModelBuilder`. No base class and no inheritance.
- `Candle` is a `NamedTuple` with the fields `timestamp` (int, ms), `open`, `high`, `low`, `close`, `volume` (float), in that order.
- No API keys in this epic. Candles are public.
- `pyproject.toml` runs pyright in strict mode over `src/` only, with `stubPath = "typings"`. ccxt ships no type information, so a local stub in `typings/ccxt/__init__.pyi` declares the parts used (verified while writing this plan: without the stub, strict mode reports `reportMissingTypeStubs` and `reportUnknownMemberType`; with it, 0 errors).
- `uv run pytest` must stay offline. Tests that touch the network are marked `network` and deselected by default.
- A pytest exit code of 5 (no tests collected) counts as a pass in the lefthook hooks.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **Unsupported interval** (for example `3h`, or `1W`, which Bitvavo doesn't offer): expect a `ValueError` that names the interval and the exchange's supported intervals, not a ccxt error from deep inside the request. Pinned in Task 2, `test_fetch_closed_candles_rejects_unsupported_interval`.
- **A candle that closes exactly at "now"** (`timestamp + interval == now`): it has closed, so it's included. Pinned in Task 2, `test_fetch_closed_candles_includes_candle_closing_exactly_now`.
- **Rows out of order from the exchange:** returned sorted by timestamp. Pinned in Task 2, `test_fetch_closed_candles_sorts_by_timestamp`.
- **An empty response** (nothing after `since_ms`): returns `[]` and doesn't raise. Pinned in Task 2, `test_fetch_closed_candles_returns_empty_list_when_exchange_has_nothing`.
- **Unknown exchange ID** (for example a typo, `bitvavoo`): a `ValueError` naming it, raised when the object is built rather than on first use. Pinned in Task 2, `test_ccxt_exchange_rejects_unknown_exchange_id`.

---

## Task 1: Pre-push quality gate

**Files:**
- Modify: `.lefthook.yml`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`: `network` marker, deselected by default)
- Modify: `CLAUDE.md` (Commands and "Development workflow and pull requests" sections)

**Interfaces:**
- Produces: the `network` pytest marker, used by Task 2's smoke test, and a `pre-push` hook that runs the full suite.

- [ ] **Step 1: Add the `network` marker and deselect it by default**

In `pyproject.toml`, `[tool.pytest.ini_options]`, change `addopts` and `markers` to:

```toml
addopts = [                                  # Additional command-line options
    "-v",                                    # Verbose output
    "--strict-markers",                      # Error on unknown markers
    "--cov=fart",                            # Measure coverage for this package
    "--cov-report=term-missing",             # Show missing lines in terminal
    "-m", "not network",                     # Stay offline unless asked (`-m network`)
]
markers = [                                  # Custom markers
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "network: calls a live exchange API (deselected by default; run with '-m network')",
]
```

- [ ] **Step 2: Verify that a later `-m` overrides the default**

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `77 passed` (nothing is marked `network` yet).

Run: `uv run pytest -m network -q 2>&1 | tail -1; uv run pytest -m network -q >/dev/null 2>&1; echo "exit $?"`
Expected: `77 deselected`, then `exit 5`. The command-line `-m` replaces the `addopts` one, and exit code 5 is what the hooks must treat as a pass.

- [ ] **Step 3: Add the pre-push gate and the exit-5 handling to `.lefthook.yml`**

Replace the whole file with:

```yaml
# There is no CI, so lefthook is the gate. pre-commit only sees staged files:
# a commit that touches only pyproject.toml, uv.lock or .mcp.json runs no tests
# and no type check there. pre-push closes that gap by running the full suite
# on the whole repository before anything leaves the machine. Skip a single
# push with `git push --no-verify` when you mean to.
#
# pytest exits with 5 when it collects no tests; that is not a failure here.
pre-push:
  commands:
    check:
      run: >-
        uv run ruff format --check .
        && uv run ruff check .
        && uv run pyright
        && { uv run pytest; code=$?; [ "$code" = "5" ] && exit 0 || exit "$code"; }

pre-commit:
  commands:
    clean-jupyter:
      priority: 2
      files: git ls-files
      glob: '*.ipynb'
      run: uv run jupyter nbconvert --clear-output --inplace {staged_files}
      stage_fixed: true
    format:
      priority: 2
      files: git ls-files
      glob: '*.{py,ipynb}'
      run: uv run ruff format {staged_files}
      stage_fixed: true
    ruff:
      priority: 2
      files: git ls-files
      glob: '*.py'
      run: uv run ruff check {staged_files} --fix
    test:
      priority: 2
      files: git ls-files
      glob: '*.{py,ipynb}'
      run: uv run pytest; code=$?; [ "$code" = "5" ] && exit 0 || exit "$code"
    typecheck:
      priority: 2
      files: git ls-files
      glob: '*.py'
      run: uv run pyright
```

- [ ] **Step 4: Install the hooks and run the gate by hand**

Run: `lefthook install && ls .git/hooks/pre-push`
Expected: `.git/hooks/pre-push` exists.

Run: `lefthook run pre-push`
Expected: the `check` command passes (`63 files already formatted`, `All checks passed!`, `0 errors`, `77 passed`).

- [ ] **Step 5: Document the gate in `CLAUDE.md`**

In the Commands section, replace the sentence starting "Pre-commit hooks are managed by `lefthook`" with:

```markdown
Hooks are managed by `lefthook` (`.lefthook.yml`) and stand in for CI, which this project doesn't have. On every commit, notebooks get their outputs stripped, staged Python files get `ruff format` + `ruff check --fix`, then `pyright` and `pytest` run. On every push, the full suite runs on the whole repository (`ruff format --check .`, `ruff check .`, `pyright`, `pytest`), because pre-commit only sees staged files and a commit touching only `pyproject.toml`, `uv.lock` or `.mcp.json` would otherwise skip tests and type checks. `git push --no-verify` skips the gate for one push; use it deliberately, never by default. Tests marked `network` call a live exchange and are deselected by default; run them with `uv run pytest -m network`.
```

In "Development workflow and pull requests", replace the bullet "There's no CI. Lefthook runs `ruff` and `pyright` on each commit, and `uv run pytest` must pass before each task commit." with:

```markdown
- There's no CI. Lefthook is the gate: pre-commit checks staged files and runs the tests, and pre-push runs the full suite on the whole repository (see Commands).
```

- [ ] **Step 6: Commit**

```bash
git add .lefthook.yml pyproject.toml CLAUDE.md
git commit -m "chore: add pre-push quality gate in place of CI"
```

---

## Task 2: Exchange layer over ccxt

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add ccxt`)
- Create: `typings/ccxt/__init__.pyi`
- Create: `src/fart/exchange/__init__.py`
- Create: `src/fart/exchange/candle.py`
- Create: `src/fart/exchange/exchange.py`
- Create: `src/fart/exchange/ccxt_exchange.py`
- Test: `tests/exchange/test_ccxt_exchange.py`
- Test: `tests/exchange/test_ccxt_exchange_network.py`

**Interfaces:**
- Consumes: the `network` marker from Task 1.
- Produces (used by story #51's `CandleCache` and story #52's server):
  - `fart.exchange.Candle(timestamp: int, open: float, high: float, low: float, close: float, volume: float)`, a `NamedTuple`
  - `fart.exchange.Exchange`, a `Protocol` with `has_market(self, market: str) -> bool` and `fetch_closed_candles(self, market: str, interval: str, since_ms: int, limit: int) -> list[Candle]`
  - `fart.exchange.CcxtExchange(exchange_id: str = "bitvavo", client: ccxt.Exchange | None = None, clock: Callable[[], int] = <now in ms>)`

- [ ] **Step 1: Add the dependency and the type stub**

Run: `uv add ccxt`
Expected: `pyproject.toml` gains `"ccxt>=4.5.85"` (or the current version) and `uv.lock` updates.

Create `typings/ccxt/__init__.pyi`:

```python
from typing import Any

class BaseError(Exception): ...
class ExchangeError(BaseError): ...
class NetworkError(BaseError): ...

class Exchange:
    id: str
    timeframes: dict[str, str]
    def __init__(self, config: dict[str, Any] = ...) -> None: ...
    def load_markets(
        self, reload: bool = ..., params: dict[str, Any] = ...
    ) -> dict[str, Any]: ...
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = ...,
        since: int | None = ...,
        limit: int | None = ...,
        params: dict[str, Any] = ...,
    ) -> list[list[float]]: ...
    @staticmethod
    def parse_timeframe(timeframe: str) -> int: ...

exchanges: list[str]

def __getattr__(name: str) -> type[Exchange]: ...
```

- [ ] **Step 2: Write `Candle` and the `Exchange` Protocol**

Create `src/fart/exchange/candle.py`:

```python
from typing import NamedTuple


class Candle(NamedTuple):
    """
    One OHLCV candle. The field order matches the cached CSV columns
    (`Timestamp,Open,High,Low,Close,Volume`).

    A NamedTuple rather than a pydantic model: the 1m cache holds millions
    of rows, and per-row validation would make loading it slow.

    """

    timestamp: int  # Candle open time, milliseconds since the epoch (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
```

Create `src/fart/exchange/exchange.py`:

```python
from typing import Protocol

from fart.exchange.candle import Candle


class Exchange(Protocol):
    """
    Structural interface for market data from an exchange.

    Any object with these methods satisfies this Protocol -- no inheritance
    required. `CcxtExchange` (`ccxt_exchange.py`) is the implementation;
    tests use an in-memory fake. Only what has a caller is declared here:
    balances and orders are added when the paper-trading and live-trading
    epics need them.

    Markets use ccxt's symbol format, e.g. `BTC/EUR`.

    """

    def has_market(self, market: str) -> bool: ...

    def fetch_closed_candles(
        self, market: str, interval: str, since_ms: int, limit: int
    ) -> list[Candle]:
        """
        Return up to `limit` closed candles starting at `since_ms`, sorted
        by timestamp. The candle that is still forming is never included.

        """
        ...
```

- [ ] **Step 3: Write the failing unit tests**

Create `tests/exchange/test_ccxt_exchange.py`:

```python
from typing import Any

import pytest

from fart.exchange import Candle, CcxtExchange

HOUR_MS = 3_600_000


class StubClient:
    """Stands in for a ccxt client: canned markets, timeframes and rows."""

    def __init__(self, rows: list[list[float]] | None = None) -> None:
        self.timeframes = {"1m": "1m", "1h": "1h", "1d": "1d"}
        self.rows = rows or []
        self.load_markets_calls = 0
        self.fetch_calls: list[dict[str, Any]] = []

    def load_markets(self) -> dict[str, Any]:
        self.load_markets_calls += 1
        return {"BTC/EUR": {}, "ETH/EUR": {}}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        self.fetch_calls.append(
            {"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit}
        )
        return self.rows


def _row(timestamp: int, close: float = 100.0) -> list[float]:
    return [timestamp, close, close + 1, close - 1, close, 2.5]


def _exchange(client: StubClient, now_ms: int) -> CcxtExchange:
    return CcxtExchange(client=client, clock=lambda: now_ms)


def test_has_market_is_true_for_listed_market() -> None:
    exchange = _exchange(StubClient(), now_ms=0)

    assert exchange.has_market("BTC/EUR")
    assert not exchange.has_market("DOGE/EUR")


def test_markets_are_loaded_only_once() -> None:
    client = StubClient()
    exchange = _exchange(client, now_ms=0)

    exchange.has_market("BTC/EUR")
    exchange.has_market("ETH/EUR")
    exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert client.load_markets_calls == 1


def test_fetch_closed_candles_passes_request_through() -> None:
    client = StubClient()
    exchange = _exchange(client, now_ms=10 * HOUR_MS)

    exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=HOUR_MS, limit=500)

    assert client.fetch_calls == [
        {"symbol": "BTC/EUR", "timeframe": "1h", "since": HOUR_MS, "limit": 500}
    ]


def test_fetch_closed_candles_returns_candles() -> None:
    client = StubClient(rows=[_row(0, close=100.0)])
    exchange = _exchange(client, now_ms=2 * HOUR_MS)

    candles = exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert candles == [Candle(0, 100.0, 101.0, 99.0, 100.0, 2.5)]
    assert isinstance(candles[0].timestamp, int)


def test_fetch_closed_candles_drops_forming_candle() -> None:
    client = StubClient(rows=[_row(0), _row(HOUR_MS), _row(2 * HOUR_MS)])
    # 2h30m: the 2h candle is still forming until 3h.
    exchange = _exchange(client, now_ms=2 * HOUR_MS + HOUR_MS // 2)

    candles = exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert [c.timestamp for c in candles] == [0, HOUR_MS]


def test_fetch_closed_candles_includes_candle_closing_exactly_now() -> None:
    client = StubClient(rows=[_row(0), _row(HOUR_MS)])
    exchange = _exchange(client, now_ms=2 * HOUR_MS)

    candles = exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert [c.timestamp for c in candles] == [0, HOUR_MS]


def test_fetch_closed_candles_sorts_by_timestamp() -> None:
    client = StubClient(rows=[_row(2 * HOUR_MS), _row(0), _row(HOUR_MS)])
    exchange = _exchange(client, now_ms=10 * HOUR_MS)

    candles = exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert [c.timestamp for c in candles] == [0, HOUR_MS, 2 * HOUR_MS]


def test_fetch_closed_candles_returns_empty_list_when_exchange_has_nothing() -> None:
    exchange = _exchange(StubClient(rows=[]), now_ms=10 * HOUR_MS)

    assert exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10) == []


def test_fetch_closed_candles_rejects_unknown_market() -> None:
    exchange = _exchange(StubClient(), now_ms=0)

    with pytest.raises(ValueError, match="DOGE/EUR.*bitvavo"):
        exchange.fetch_closed_candles("DOGE/EUR", "1h", since_ms=0, limit=10)


def test_fetch_closed_candles_rejects_unsupported_interval() -> None:
    exchange = _exchange(StubClient(), now_ms=0)

    with pytest.raises(ValueError, match=r"3h.*1m, 1h, 1d"):
        exchange.fetch_closed_candles("BTC/EUR", "3h", since_ms=0, limit=10)


def test_ccxt_exchange_rejects_unknown_exchange_id() -> None:
    with pytest.raises(ValueError, match="bitvavoo"):
        CcxtExchange(exchange_id="bitvavoo")


def test_ccxt_exchange_builds_real_client_for_known_id() -> None:
    # Building the client makes no network call; markets load lazily.
    exchange = CcxtExchange(exchange_id="bitvavo")

    assert exchange.exchange_id == "bitvavo"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/exchange/test_ccxt_exchange.py -q 2>&1 | tail -3`
Expected: collection error, `ModuleNotFoundError: No module named 'fart.exchange'` (or `ImportError` for `CcxtExchange`).

- [ ] **Step 5: Implement `CcxtExchange` and the package exports**

Create `src/fart/exchange/ccxt_exchange.py`:

```python
import time
from collections.abc import Callable
from typing import Any

import ccxt

from fart.exchange.candle import Candle


def _now_ms() -> int:
    return int(time.time() * 1000)


class CcxtExchange:
    """
    `Exchange` implementation backed by ccxt -- the only module in the
    project that imports ccxt, so the exchange stays swappable.

    `exchange_id` names the ccxt exchange (default `bitvavo`). `client` and
    `clock` are injectable for tests: `clock` returns the current time in
    milliseconds and decides which candles have closed.

    """

    def __init__(
        self,
        exchange_id: str = "bitvavo",
        client: ccxt.Exchange | None = None,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        if client is None:
            if exchange_id not in ccxt.exchanges:
                raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")
            exchange_class: type[ccxt.Exchange] = getattr(ccxt, exchange_id)
            client = exchange_class({"enableRateLimit": True})

        self.exchange_id = exchange_id
        self._client = client
        self._clock = clock
        self._markets: dict[str, Any] | None = None

    def has_market(self, market: str) -> bool:
        return market in self._load_markets()

    def fetch_closed_candles(
        self, market: str, interval: str, since_ms: int, limit: int
    ) -> list[Candle]:
        self._validate(market, interval)

        rows = self._client.fetch_ohlcv(market, interval, since=since_ms, limit=limit)
        interval_ms = ccxt.Exchange.parse_timeframe(interval) * 1000
        now_ms = self._clock()

        candles = [
            Candle(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in rows
        ]
        # A candle has closed once its period has ended; the last row ccxt
        # returns is usually the one still forming.
        closed = [c for c in candles if c.timestamp + interval_ms <= now_ms]

        return sorted(closed, key=lambda candle: candle.timestamp)

    def _validate(self, market: str, interval: str) -> None:
        if not self.has_market(market):
            raise ValueError(
                f"Market '{market}' not found on exchange '{self.exchange_id}'"
            )
        if interval not in self._client.timeframes:
            supported = ", ".join(self._client.timeframes)
            raise ValueError(
                f"Interval '{interval}' is not supported by exchange "
                f"'{self.exchange_id}' (supported: {supported})"
            )

    def _load_markets(self) -> dict[str, Any]:
        if self._markets is None:
            self._markets = self._client.load_markets()
        return self._markets
```

Create `src/fart/exchange/__init__.py`:

```python
from fart.exchange.candle import Candle
from fart.exchange.ccxt_exchange import CcxtExchange
from fart.exchange.exchange import Exchange

__all__ = ["Candle", "CcxtExchange", "Exchange"]
```

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `uv run pytest tests/exchange/test_ccxt_exchange.py -q 2>&1 | grep -E "passed|failed|error"`
Expected: `12 passed`.

- [ ] **Step 7: Write the live smoke test**

Create `tests/exchange/test_ccxt_exchange_network.py`:

```python
import time

import pytest

from fart.exchange import CcxtExchange

HOUR_MS = 3_600_000


@pytest.mark.network
def test_bitvavo_returns_recent_closed_btc_eur_candles() -> None:
    exchange = CcxtExchange(exchange_id="bitvavo")
    now_ms = int(time.time() * 1000)

    candles = exchange.fetch_closed_candles(
        "BTC/EUR", "1h", since_ms=now_ms - 6 * HOUR_MS, limit=10
    )

    assert candles, "expected at least one closed 1h candle in the last 6 hours"
    timestamps = [c.timestamp for c in candles]
    assert timestamps == sorted(timestamps)
    assert all(t + HOUR_MS <= now_ms for t in timestamps)
    assert all(c.low <= c.close <= c.high for c in candles)
```

- [ ] **Step 8: Run the smoke test against the live exchange, then the default suite**

Run: `uv run pytest -m network -q 2>&1 | grep -E "passed|failed|error"`
Expected: `1 passed, 89 deselected` (needs network access).

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: `89 passed, 1 deselected`, offline.

- [ ] **Step 9: Type-check, lint, and confirm ccxt stays behind the interface**

Run: `uv run pyright 2>&1 | grep -E "error|errors"`
Expected: `0 errors, 0 warnings, 0 informations`.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: `All checks passed!` and every file already formatted.

Run: `git grep -n "import ccxt" -- src tests`
Expected: only `src/fart/exchange/ccxt_exchange.py`.

- [ ] **Step 10: Record the confirmed findings in the PRD**

Issue #50 asks for the ccxt stop-loss coverage and the fee tier to be settled in the PRD, where both are tagged as assumptions. In `docs/product/mcp-trading-agent-prd.md`:

- Section 6, "Why sub-30m is excluded": replace `(🔶 **Assumption:** to be confirmed against the actual Bitvavo fee tier)` with `(confirmed 2026-10-07 from ccxt's Bitvavo fee table: 0.25% taker, 0.15% maker at the entry tier; the operator's own tier still needs an authenticated check)`.
- Section 9, Dependencies, the ccxt bullet: replace `(🔶 **Assumption:** to be verified in A1 and F2)` with `(confirmed 2026-10-07: a `stopLossPrice` parameter on `create_order` maps to Bitvavo's native `stopLoss`/`stopLossLimit` orders, placed as a separate order after the entry fills; Bitvavo also requires an `operatorId` on every order)`.
- Section 10, the assumptions list: remove the "Bitvavo fee tier" and "ccxt coverage" bullets.

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml uv.lock typings/ccxt/__init__.pyi src/fart/exchange tests/exchange docs/product/mcp-trading-agent-prd.md
git commit -m "feat: add exchange layer fetching closed candles through ccxt"
```

- [ ] **Step 12: Push (runs the pre-push gate)**

Run: `git push`
Expected: lefthook's `check` passes, then the push goes through.
