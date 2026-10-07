import threading
from pathlib import Path

import pytest

from fartt.candle_cache import CandleCache, interval_to_ms
from fartt.exchange import Candle
from tests.fakes import HOUR_MS, FakeExchange, candle

HEADER = "Timestamp,Open,High,Low,Close,Volume\n"


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


def _write(path: Path, candles: list[Candle]) -> None:
    lines = [HEADER] + [",".join(str(v) for v in c) + "\n" for c in candles]
    path.write_text("".join(lines))


def _timestamps(path: Path) -> list[int]:
    return [int(line.split(",")[0]) for line in path.read_text().splitlines()[1:]]


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1m", 60_000),
        ("1h", HOUR_MS),
        ("4h", 4 * HOUR_MS),
        ("1d", 24 * HOUR_MS),
        ("1W", 7 * 24 * HOUR_MS),
        ("1M", 30 * 24 * HOUR_MS),
    ],
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
    exchange = FakeExchange(
        [candle(h) for h in range(6)], now_ms=5 * HOUR_MS + HOUR_MS // 2
    )
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


def test_update_drops_a_torn_last_line_and_refetches_it(tmp_path: Path) -> None:
    # A hard kill mid-append leaves half a line and no trailing newline.
    exchange = FakeExchange([candle(h) for h in range(4)], now_ms=4 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(0), candle(1)])
    with cache.filepath.open("a") as file:
        file.write(f"{2 * HOUR_MS},100.0,101")

    assert cache.update() == 2
    assert cache.latest(10) == [candle(h) for h in range(4)]


def test_update_completes_a_valid_last_line_missing_its_newline(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    _write(cache.filepath, [candle(0), candle(1)])
    cache.filepath.write_text(cache.filepath.read_text().rstrip("\n"))

    assert cache.update() == 1
    assert cache.latest(10) == [candle(h) for h in range(3)]


def test_update_completes_a_header_missing_its_newline(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(2)], now_ms=2 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.filepath.write_text(HEADER.rstrip("\n"))

    assert cache.update() == 2
    assert cache.filepath.read_text().splitlines()[0] == HEADER.rstrip("\n")
    assert cache.latest(10) == [candle(0), candle(1)]


def test_a_second_updater_waits_for_the_first(tmp_path: Path) -> None:
    # `fartt download` and the MCP server share the file; two writers at
    # once would append the same candles twice.
    exchange = FakeExchange([candle(h) for h in range(5)], now_ms=5 * HOUR_MS)
    first = _cache(exchange, tmp_path)
    second = _cache(exchange, tmp_path)
    updates = first.iter_update()
    assert next(updates) == 2  # first is mid-update, holding the file

    second_result: list[int] = []
    waiter = threading.Thread(target=lambda: second_result.append(second.update()))
    waiter.start()
    waiter.join(timeout=0.3)
    assert waiter.is_alive(), "second updater ran while the first held the file"

    assert sum(updates) == 3
    waiter.join(timeout=5)
    assert second_result == [0]
    assert _timestamps(first.filepath) == [h * HOUR_MS for h in range(5)]
