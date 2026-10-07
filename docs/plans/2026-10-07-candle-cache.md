# Candle Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `fartt download` and the coming MCP server one shared candle cache that fetches through the exchange layer (story #51), replacing `Downloader` and its direct Bitvavo calls.

**Architecture:** `fartt.candle_cache.CandleCache` owns one market and interval's CSV file. `iter_update()` pages through closed candles from the `Exchange` interface, appends each batch to the CSV and yields its size. `update()` runs that to the end, and `latest(count)` returns the newest cached candles. `fartt download` builds a `CcxtExchange` and a `CandleCache` and wraps `iter_update()` in tqdm. `downloader.py` is deleted.

**Tech Stack:** Python 3.11+, ccxt (through `fartt.exchange`), Typer, tqdm, tabulate, loguru, pytest, pyright (strict), uv, Task.

**Spec:** `docs/specs/2026-10-07-server-foundation-design.md` (section "2. Candle cache and `fartt download`"), plus the two points the story #50 review left for this plan (below).

## Global Constraints

- Market symbols use ccxt's format, `BTC/EUR`, on the CLI and in code. Cache file names keep `BTC-EUR`: `get_data_filepath` converts `/` to `-`, so `BTC/EUR` + `1h` → `assets/BTC-EUR-1h.csv`, and a notebook passing `BTC-EUR` gets the same path.
- The CSV format doesn't change: header `Timestamp,Open,High,Low,Close,Volume` (from `fartt.constants`), one candle per line, timestamps in milliseconds, a trailing newline. Existing cache files stay valid.
- The cache writes only by **appending**, except for replacing its own last line (see "Settled here"). It never rewrites the whole file.
- The default history start is 2019-03-09 (`1_552_089_600_000` ms), Bitvavo's launch, as a constructor parameter rather than something the exchange layer knows.
- No API keys: candles are public. `fartt download` stops reading `BITVAVO_API_KEY`/`BITVAVO_API_SECRET`.
- The cache depends on the `Exchange` Protocol, never on `CcxtExchange` or ccxt. Tests use an in-memory fake.
- `task check` must pass before each commit; lefthook runs it on push.
- Never reference "superpowers" in code, file paths or directory structure.

## Settled here (left open by the story #50 review)

- **Gap or present?** An empty batch from `fetch_closed_candles` can mean a gap in the exchange's history or that nothing newer has closed yet. The cache tells them apart with its own injectable clock: it pages only while `since_ms` is at or before the newest **closed** period's open time, `floor(now / interval) * interval - interval`. An empty batch inside that range is a gap, so it skips ahead one batch window (`since_ms += batch_limit * interval_ms`). The loop then ends once `since_ms` passes the present. If the two clocks disagree, the worst case is that the cache stops early, and the next update resumes from the file, since a skip never persists.
- **Short batches and ccxt's 1440 cap.** The cache pages from the last returned timestamp plus one interval, never from `since_ms + limit * interval` (as the `Exchange` docstring now requires).
- **A candle the exchange amends after it closes.** Each update starts by re-fetching the last cached candle (`since_ms` = its timestamp). If the exchange's version differs, the cache replaces the file's last line with it. That's the one in-place write, and it truncates only the final line. Re-fetching it never duplicates it, and it doesn't count as a new candle.
- **`fartt download` on a single-command Typer app.** With one command and no callback, Typer runs that command as the whole CLI, so `fartt download` fails today with "Got unexpected extra argument (download)". An `@app.callback()` makes `download` a real subcommand, as the README and issue #51 use it.

## Review Focus

- **A cache file holding only the header** (for example from an interrupted first run): expect it to be treated as empty and filled from the history start, not to crash. Pinned in Task 1, `test_update_treats_header_only_file_as_empty`.
- **A gap longer than one batch window** (the exchange was down for days; ccxt asks Bitvavo for `[since, since + limit × interval)`, so such a window comes back empty): expect every candle after the gap to be fetched. Pinned in Task 1, `test_update_skips_gaps_longer_than_one_batch`.
- **Running `update()` twice in a row with nothing new:** expect 0 new candles and a byte-identical file. Pinned in Task 1, `test_update_twice_adds_nothing_the_second_time`.
- **An existing 1m cache of ~3 million rows:** resuming must not read the whole file to find the last candle. Pinned in Task 1, `test_update_reads_only_the_file_tail_to_resume` (asserts the tail reader reads a bounded number of bytes).
- **`fartt download --market BTC-EUR`** (the old format, still in muscle memory): expect a clear error naming `BTC/EUR`, not a stack trace. Pinned in Task 2, `test_download_rejects_unknown_market_with_hint`.

---

## Task 1: `CandleCache`

**Files:**
- Create: `src/fartt/candle_cache.py`
- Modify: `src/fartt/utils.py` (`get_data_filepath` converts `/` to `-`)
- Create: `tests/fakes.py`
- Create: `tests/test_candle_cache.py`
- Modify: `tests/utils/test_get_data_filepath.py`

**Interfaces:**
- Consumes: `fartt.exchange.Candle` and `fartt.exchange.Exchange` (story #50).
- Produces (used by Task 2 and story #52):
  - `fartt.candle_cache.CandleCache(exchange: Exchange, assets_dir: Path, market: str, interval: str, history_start_ms: int = DEFAULT_HISTORY_START_MS, batch_limit: int = DEFAULT_BATCH_LIMIT, clock: Callable[[], int] = <now in ms>)`
    - `.filepath: Path`
    - `.iter_update() -> Iterator[int]`: yields the number of new candles appended per batch (batches with no new candles yield nothing)
    - `.update() -> int`: total new candles
    - `.latest(count: int) -> list[Candle]`: up to `count` newest cached candles, oldest first
  - `fartt.candle_cache.interval_to_ms(interval: str) -> int`
  - `fartt.candle_cache.DEFAULT_HISTORY_START_MS = 1_552_089_600_000`, `DEFAULT_BATCH_LIMIT = 1000`
  - `tests/fakes.py::FakeExchange(candles: list[Candle], now_ms: int, markets: set[str] = {"BTC/EUR"})` with `.fetch_calls: list[int]` (the `since_ms` of each call)

- [ ] **Step 1: Write the fake exchange**

Create `tests/fakes.py`:

```python
from fartt.candle_cache import interval_to_ms
from fartt.exchange import Candle

HOUR_MS = 3_600_000


class FakeExchange:
    """
    In-memory `Exchange`: serves `candles` the way `CcxtExchange` over
    Bitvavo does -- closed candles only (judged against `now_ms`), sorted,
    from the window `[since_ms, since_ms + limit * interval)`. ccxt sends
    Bitvavo exactly that window, so a gap longer than it comes back empty.

    """

    def __init__(
        self,
        candles: list[Candle],
        now_ms: int,
        markets: set[str] | None = None,
    ) -> None:
        self.candles = candles
        self.now_ms = now_ms
        self.markets = markets if markets is not None else {"BTC/EUR"}
        self.fetch_calls: list[int] = []

    def has_market(self, market: str) -> bool:
        return market in self.markets

    def fetch_closed_candles(
        self, market: str, interval: str, since_ms: int, limit: int
    ) -> list[Candle]:
        if not self.has_market(market):
            raise ValueError(f"Market '{market}' not found on exchange 'fake'")
        self.fetch_calls.append(since_ms)
        interval_ms = interval_to_ms(interval)
        window_end_ms = since_ms + limit * interval_ms
        return sorted(
            (
                c
                for c in self.candles
                if since_ms <= c.timestamp < window_end_ms
                and c.timestamp + interval_ms <= self.now_ms
            ),
            key=lambda c: c.timestamp,
        )


def candle(hour: int, close: float = 100.0) -> Candle:
    """An hourly candle opening `hour` hours after the epoch."""
    return Candle(hour * HOUR_MS, close, close + 1, close - 1, close, 2.5)
```

- [ ] **Step 2: Write the failing filename tests**

Append to `tests/utils/test_get_data_filepath.py`:

```python
def test_get_data_filepath_converts_ccxt_symbol_to_file_name() -> None:
    filepath = get_data_filepath(
        data_dir=Path("/tmp/fartt-test-data"),
        market="BTC/EUR",
        interval="1h",
    )

    assert filepath == Path("/tmp/fartt-test-data/BTC-EUR-1h.csv")
```

Run: `uv run pytest tests/utils/test_get_data_filepath.py -q 2>&1 | tail -1`
Expected: `1 failed, 2 passed` (the path contains a `BTC/EUR` subdirectory).

- [ ] **Step 3: Convert the market symbol in `get_data_filepath`**

In `src/fartt/utils.py`, change `get_data_filepath`'s docstring market example and body:

```python
def get_data_filepath(data_dir: Path, market: str, interval: str) -> Path:
    """
    Get the file path for a candle data file.

    Parameters
    ----------
    - data_dir (Path): Path to the directory containing data files.
    - market (str): Market name in ccxt's format (e.g., 'BTC/EUR'). The
      file name uses 'BTC-EUR', so the older dash form maps to the same file.
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').

    Returns
    -------
    - Path: Path to the candle data file.

    """
    return data_dir / f"{market.replace('/', '-')}-{interval}.csv"
```

Run: `uv run pytest tests/utils/test_get_data_filepath.py -q 2>&1 | tail -1`
Expected: `3 passed`.

- [ ] **Step 4: Write the failing cache tests**

Create `tests/test_candle_cache.py`:

```python
from pathlib import Path

import pytest

from fartt.candle_cache import CandleCache, interval_to_ms
from fartt.exchange import Candle
from tests.fakes import HOUR_MS, FakeExchange, candle

HEADER = "Timestamp,Open,High,Low,Close,Volume\n"


def _cache(
    exchange: FakeExchange, tmp_path: Path, batch_limit: int = 2
) -> CandleCache:
    return CandleCache(
        exchange=exchange,
        assets_dir=tmp_path,
        market="BTC/EUR",
        interval="1h",
        history_start_ms=0,
        batch_limit=batch_limit,
        clock=lambda: exchange.now_ms,
    )


def _write(path: Path, candles: list[Candle]) -> None:
    lines = [HEADER] + [",".join(str(v) for v in c) + "\n" for c in candles]
    path.write_text("".join(lines))


def _timestamps(path: Path) -> list[int]:
    return [int(line.split(",")[0]) for line in path.read_text().splitlines()[1:]]


@pytest.mark.parametrize(
    ("interval", "expected"),
    [("1m", 60_000), ("1h", HOUR_MS), ("4h", 4 * HOUR_MS), ("1d", 24 * HOUR_MS),
     ("1W", 7 * 24 * HOUR_MS), ("1M", 30 * 24 * HOUR_MS)],
)
def test_interval_to_ms(interval: str, expected: int) -> None:
    assert interval_to_ms(interval) == expected


def test_interval_to_ms_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError, match="3x"):
        interval_to_ms("3x")


def test_filepath_uses_dash_form_of_market(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)

    assert cache.filepath == tmp_path / "BTC-EUR-1h.csv"


def test_update_fills_empty_cache_from_history_start(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    cache = _cache(exchange, tmp_path)

    batches = list(cache.iter_update())

    assert batches == [2, 2, 1]
    assert exchange.fetch_calls[0] == 0
    assert cache.filepath.read_text().startswith(HEADER)
    assert _timestamps(cache.filepath) == [h * HOUR_MS for h in range(5)]


def test_update_never_writes_the_forming_candle(tmp_path: Path) -> None:
    # 5h30m: the 5h candle is still forming.
    exchange = FakeExchange([candle(h) for h in range(6)], now_ms=5 * HOUR_MS + HOUR_MS // 2)
    cache = _cache(exchange, tmp_path)

    assert cache.update() == 5
    assert _timestamps(cache.filepath)[-1] == 4 * HOUR_MS


def test_update_resumes_from_last_cached_candle(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(h) for h in range(3)])

    assert cache.update() == 2
    # Starts by re-fetching the last cached candle, never from the start.
    assert exchange.fetch_calls[0] == 2 * HOUR_MS
    assert _timestamps(cache.filepath) == [h * HOUR_MS for h in range(5)]


def test_update_appends_without_rewriting_existing_lines(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(h) for h in range(3)])
    before = cache.filepath.read_text()

    cache.update()

    assert cache.filepath.read_text().startswith(before)


def test_update_replaces_last_candle_when_exchange_amended_it(tmp_path: Path) -> None:
    exchange = FakeExchange(
        [candle(0), candle(1), candle(2, close=150.0), candle(3)], now_ms=4 * HOUR_MS
    )
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(0), candle(1), candle(2, close=120.0)])

    assert cache.update() == 1
    assert cache.latest(2) == [candle(2, close=150.0), candle(3)]
    assert _timestamps(cache.filepath) == [0, HOUR_MS, 2 * HOUR_MS, 3 * HOUR_MS]


def test_update_twice_adds_nothing_the_second_time(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    before = cache.filepath.read_bytes()

    assert cache.update() == 0
    assert cache.filepath.read_bytes() == before


def test_update_skips_gaps_longer_than_one_batch(tmp_path: Path) -> None:
    # Hours 3-9 missing: longer than one 2-candle batch window.
    hours = [0, 1, 2, 10, 11]
    exchange = FakeExchange([candle(h) for h in hours], now_ms=12 * HOUR_MS)
    cache = _cache(exchange, tmp_path)

    assert cache.update() == 5
    assert _timestamps(cache.filepath) == [h * HOUR_MS for h in hours]


def test_update_stops_at_present_when_nothing_newer_closed(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path, batch_limit=10)
    cache.update()
    exchange.fetch_calls.clear()

    assert cache.update() == 0
    assert exchange.fetch_calls == [2 * HOUR_MS]


def test_update_treats_header_only_file_as_empty(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.filepath.write_text(HEADER)

    assert cache.update() == 3
    assert cache.filepath.read_text().count("Timestamp") == 1


def test_update_reads_only_the_file_tail_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exchange = FakeExchange([candle(h) for h in range(2001)], now_ms=2001 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(h) for h in range(2000)])
    size = cache.filepath.stat().st_size
    read_sizes: list[int] = []
    real_open = open

    class CountingFile:
        def __init__(self, f):
            self._f = f

        def read(self, n: int = -1) -> bytes:
            data = self._f.read(n)
            read_sizes.append(len(data))
            return data

        def __getattr__(self, name: str):
            return getattr(self._f, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._f.__exit__(*args)

    def counting_open(path, mode="r", *args, **kwargs):
        f = real_open(path, mode, *args, **kwargs)
        return CountingFile(f) if "b" in mode and "r" in mode else f

    monkeypatch.setattr("fartt.candle_cache.open", counting_open, raising=False)

    assert cache.update() == 1
    assert sum(read_sizes) < size // 10


def test_latest_returns_newest_candles_oldest_first(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)
    _write(cache.filepath, [candle(h) for h in range(5)])

    assert cache.latest(2) == [candle(3), candle(4)]
    assert cache.latest(10) == [candle(h) for h in range(5)]


def test_latest_on_missing_file_is_empty(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)

    assert cache.latest(5) == []
```

Run: `uv run pytest tests/test_candle_cache.py -q 2>&1 | grep -E "Error|error" | head -2`
Expected: `ModuleNotFoundError: No module named 'fartt.candle_cache'`.

- [ ] **Step 5: Implement `CandleCache`**

Create `src/fartt/candle_cache.py`:

```python
import time
from collections import deque
from collections.abc import Callable, Iterator
from csv import DictReader
from pathlib import Path

from fartt.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fartt.exchange import Candle, Exchange
from fartt.utils import get_data_filepath

# Bitvavo launched on 2019-03-09; nothing older exists there to fetch.
DEFAULT_HISTORY_START_MS = 1_552_089_600_000
DEFAULT_BATCH_LIMIT = 1000

HEADER = f"{TIMESTAMP},{OPEN},{HIGH},{LOW},{CLOSE},{VOLUME}\n"

# Enough bytes to hold the last line of any cache file (a line is ~60 bytes).
_TAIL_BYTES = 1024

_UNIT_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "W": 604_800_000,
    "M": 30 * 86_400_000,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def interval_to_ms(interval: str) -> int:
    """Length of a candle interval such as `1h` or `30m`, in milliseconds."""
    unit = interval[-1:]
    if unit not in _UNIT_MS or not interval[:-1].isdigit():
        raise ValueError(f"Invalid interval: {interval}")
    return int(interval[:-1]) * _UNIT_MS[unit]


class CandleCache:
    """
    One market and interval's candles, cached as a CSV file under
    `assets_dir` and kept up to date through an `Exchange`.

    The file only ever grows by appending, with one exception: each update
    re-fetches the last cached candle, and if the exchange has amended it
    since, the file's last line is replaced. Both `fartt download` and the
    MCP server read and update the cache through this class.

    `clock` returns the current time in milliseconds and decides where
    "the present" is: paging stops after the newest closed period, so an
    empty batch before that point is a gap in the exchange's history and
    is skipped.

    """

    def __init__(
        self,
        exchange: Exchange,
        assets_dir: Path,
        market: str,
        interval: str,
        history_start_ms: int = DEFAULT_HISTORY_START_MS,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        self.filepath = get_data_filepath(assets_dir, market, interval)
        self._exchange = exchange
        self._assets_dir = assets_dir
        self._market = market
        self._interval = interval
        self._interval_ms = interval_to_ms(interval)
        self._history_start_ms = history_start_ms
        self._batch_limit = batch_limit
        self._clock = clock

    def iter_update(self) -> Iterator[int]:
        """
        Fetch every closed candle newer than the cache, one batch at a time,
        and yield the number of new candles after appending each batch.

        """
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        last = self._read_last_candle()
        since_ms = last.timestamp if last is not None else self._history_start_ms
        now_ms = self._clock()
        newest_closed_ms = (now_ms // self._interval_ms - 1) * self._interval_ms

        while since_ms <= newest_closed_ms:
            batch = self._exchange.fetch_closed_candles(
                self._market, self._interval, since_ms, self._batch_limit
            )
            if not batch:
                # Nothing closed in this window although the present is
                # further on: a gap in the exchange's history. Skip it.
                since_ms += self._batch_limit * self._interval_ms
                continue

            if last is not None and batch[0].timestamp == last.timestamp:
                if batch[0] != last:
                    self._replace_last_line(batch[0])
                    last = batch[0]

            new = [c for c in batch if last is None or c.timestamp > last.timestamp]
            if new:
                self._append(new)
                last = new[-1]
                yield len(new)

            since_ms = batch[-1].timestamp + self._interval_ms

    def update(self) -> int:
        """Run `iter_update()` to the end; return the number of new candles."""
        return sum(self.iter_update())

    def latest(self, count: int) -> list[Candle]:
        """Return up to `count` of the newest cached candles, oldest first."""
        if not self.filepath.exists():
            return []
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

    def _read_tail(self) -> tuple[int, bytes]:
        # Returns (offset of the tail in the file, the tail's bytes), reading
        # at most _TAIL_BYTES so a multi-million-row cache resumes instantly.
        with open(self.filepath, "rb") as file:
            file.seek(0, 2)
            size = file.tell()
            offset = max(0, size - _TAIL_BYTES)
            file.seek(offset)
            return offset, file.read()

    def _read_last_candle(self) -> Candle | None:
        if not self.filepath.exists():
            return None
        _, tail = self._read_tail()
        last_line = tail.rstrip(b"\n").rsplit(b"\n", 1)[-1].decode("utf-8")
        if not last_line or last_line.startswith(TIMESTAMP):
            return None
        values = last_line.split(",")
        return Candle(int(values[0]), *(float(v) for v in values[1:6]))

    def _replace_last_line(self, candle: Candle) -> None:
        offset, tail = self._read_tail()
        start = offset + tail.rstrip(b"\n").rfind(b"\n") + 1
        with open(self.filepath, "rb+") as file:
            file.truncate(start)
        self._append([candle])

    def _append(self, candles: list[Candle]) -> None:
        is_new = not self.filepath.exists() or self.filepath.stat().st_size == 0
        with self.filepath.open("a", newline="", encoding="utf-8") as file:
            if is_new:
                file.write(HEADER)
            file.writelines(
                ",".join(str(value) for value in candle) + "\n" for candle in candles
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_candle_cache.py tests/utils/test_get_data_filepath.py -q 2>&1 | tail -1`
Expected: `23 passed` (20 cache tests, of which 7 are `interval_to_ms` cases, plus 3 filename tests).

- [ ] **Step 7: Run the full check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, and `111 passed, 1 deselected` (90 before, plus 21).

```bash
git add src/fartt/candle_cache.py src/fartt/utils.py tests/fakes.py tests/test_candle_cache.py tests/utils/test_get_data_filepath.py
git commit -m "feat: add candle cache shared by download and the agent"
```

---

## Task 2: `fartt download` on the cache

**Files:**
- Modify: `src/fartt/cli.py`
- Delete: `src/fartt/downloader.py`, `tests/test_downloader.py`
- Create: `tests/test_cli.py`
- Modify: `CLAUDE.md`, `README.md` (download usage)

**Interfaces:**
- Consumes: `CandleCache`, `DEFAULT_HISTORY_START_MS` (Task 1); `fartt.exchange.CcxtExchange(exchange_id)` (story #50); `tests/fakes.py::FakeExchange`, `candle` (Task 1).
- Produces: the CLI `fartt download [--exchange bitvavo] [--market BTC/EUR] [--interval 1d] [--assets-dir assets]`.

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fartt import cli
from tests.fakes import HOUR_MS, FakeExchange, candle

runner = CliRunner()


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch) -> FakeExchange:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    monkeypatch.setattr(cli, "CcxtExchange", lambda exchange_id: exchange)
    monkeypatch.setattr(cli, "DEFAULT_HISTORY_START_MS", 0)
    return exchange


def test_download_is_a_subcommand(fake_exchange: FakeExchange, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["download", "--assets-dir", str(tmp_path), "--market", "BTC/EUR", "--interval", "1h"],
    )

    assert result.exit_code == 0, result.output
    lines = (tmp_path / "BTC-EUR-1h.csv").read_text().splitlines()
    assert lines[0] == "Timestamp,Open,High,Low,Close,Volume"
    assert len(lines) == 4


def test_download_rejects_unknown_market_with_hint(
    fake_exchange: FakeExchange, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli.app,
        ["download", "--assets-dir", str(tmp_path), "--market", "BTC-EUR", "--interval", "1h"],
    )

    assert result.exit_code != 0
    assert "BTC-EUR" in result.output
    assert "BTC/EUR" in result.output
    assert not (tmp_path / "BTC-EUR-1h.csv").exists()
```

Run: `uv run pytest tests/test_cli.py -q 2>&1 | tail -1`
Expected: `2 failed` (`AttributeError: ... has no attribute 'CcxtExchange'` from the fixture).

- [ ] **Step 2: Rewrite `cli.py`**

Replace `src/fartt/cli.py` with:

```python
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from fartt.candle_cache import DEFAULT_HISTORY_START_MS, CandleCache
from fartt.exchange import CcxtExchange

app = typer.Typer(no_args_is_help=True)

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
logger.add("logs/cli.log", rotation="1 MB", level="INFO", format=LOG_FORMAT)

load_dotenv(find_dotenv())


@app.callback()
def main() -> None:
    """Fartt: Financial Analysis Real Time Trading."""
    # A callback makes every command a subcommand. Without it, Typer runs a
    # lone command as the whole CLI and `fartt download` fails with
    # "Got unexpected extra argument (download)".


@app.command()
def download(
    assets_dir: Annotated[
        str,
        typer.Option(help="Folder the candle cache lives in."),
    ] = "assets",
    exchange: Annotated[
        str,
        typer.Option(help="ccxt exchange id (e.g., 'bitvavo')."),
    ] = "bitvavo",
    interval: Annotated[
        str,
        typer.Option(
            help="Candle interval the exchange offers (e.g., '1m', '30m', '1h', '4h', '1d')."
        ),
    ] = "1d",
    market: Annotated[
        str,
        typer.Option(help="Market in ccxt's format (e.g., 'BTC/EUR', 'ETH/EUR')."),
    ] = "BTC/EUR",
) -> None:
    """Download closed candles into the local cache, resuming where it left off."""
    client = CcxtExchange(exchange_id=exchange)
    if not client.has_market(market):
        raise typer.BadParameter(
            f"Market '{market}' not found on exchange '{exchange}'. "
            "Markets use ccxt's format, e.g. 'BTC/EUR' rather than 'BTC-EUR'.",
            param_hint="--market",
        )

    cache = CandleCache(
        exchange=client,
        assets_dir=Path(assets_dir),
        market=market,
        interval=interval,
        history_start_ms=DEFAULT_HISTORY_START_MS,
    )
    configuration = {
        "exchange": exchange,
        "market": market,
        "interval": interval,
        "filepath": str(cache.filepath),
    }
    logger.info(f"\n\nFartt Downloader\n\n{tabulate(configuration.items())}\n")

    with tqdm(desc="Downloading", unit=" candles") as progress:
        for count in cache.iter_update():
            progress.update(count)


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Delete the old downloader and its test**

Run: `git rm -q src/fartt/downloader.py tests/test_downloader.py`

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -q 2>&1 | tail -1`
Expected: `2 passed`.

- [ ] **Step 5: Run it for real against Bitvavo**

Run: `uv run fartt download --assets-dir /tmp/fartt-download-check --market BTC/EUR --interval 1d 2>&1 | tail -3 && head -2 /tmp/fartt-download-check/BTC-EUR-1d.csv && wc -l /tmp/fartt-download-check/BTC-EUR-1d.csv`
Expected: a tqdm line ending in roughly 2,770 candles (one per day since 2019-03-09), the CSV header, a first row dated 2019-03-09 (`1552089600000`), and about 2,770 lines.

Run the same command again.
Expected: 0 candles downloaded, and the line count unchanged.

Run: `uv run fartt download --assets-dir assets --market BTC/EUR --interval 1h 2>&1 | tail -1`
Expected: resumes the existing `assets/BTC-EUR-1h.csv` and appends only the candles since its last row.

Run: `rm -rf /tmp/fartt-download-check`

- [ ] **Step 6: Update the documentation**

In `CLAUDE.md`, replace the paragraph starting "`fartt download` takes all arguments as options" with:

```markdown
`fartt download` takes all arguments as options (no positionals): `--assets-dir` (default `assets`), `--exchange` (a ccxt exchange id, default `bitvavo`), `--market` (ccxt's format, e.g. `BTC/EUR`, default `BTC/EUR`) and `--interval` (one the exchange offers, e.g. `1m`, `30m`, `1h`, `4h`, `1d`; default `1d`). For example `uv run fartt download --market BTC/EUR --interval 1h`. Candles are public, so no API keys are needed. It goes through `fartt.candle_cache.CandleCache` and the exchange layer: the cache is one CSV per market and interval under `assets_dir` (file names keep the dash form, `BTC-EUR-1h.csv`), and each run appends only candles newer than the last cached one, after re-fetching that last one in case the exchange amended it.
```

Also in `CLAUDE.md`'s Architecture section, replace the `**`fartt/downloader.py`**` bullet with:

```markdown
- **`fartt/candle_cache.py`** — `CandleCache`: one market and interval's candles as an append-only CSV under `assets_dir`, kept up to date through the `Exchange` interface (`fartt/exchange/`). It pages through closed candles from the last cached one, skips gaps in the exchange's history, and replaces only its own last line when the exchange amended that candle. `fartt download` and the MCP server both use it.
- **`fartt/exchange/`** — the `Exchange` Protocol (`has_market`, `fetch_closed_candles`), the `Candle` NamedTuple, and `CcxtExchange`, the only module that imports ccxt (default exchange `bitvavo`).
```

In `README.md`'s Usage section, change the example command to `uv run fartt download --assets-dir assets --market BTC/EUR --interval 1h`, and remove any sentence saying the download needs `BITVAVO_API_KEY`/`BITVAVO_API_SECRET`.

- [ ] **Step 7: Run the full check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, and `107 passed, 1 deselected` (111 from Task 1, plus 2 CLI tests, minus the 6 removed `test_downloader.py` tests).

Run: `grep -rn "python_bitvavo_api\|fartt.downloader" src tests`
Expected: only `src/fartt/core/exchange.py` (removed in story #53).

```bash
git add src/fartt/cli.py tests/test_cli.py CLAUDE.md README.md
git commit -m "feat: download candles through the cache and exchange layer"
```

- [ ] **Step 8: Push (runs the pre-push gate)**

Run: `git push`
Expected: lefthook's `task check` passes, then the push goes through.
