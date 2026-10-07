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
            {
                "time": "1970-01-01T01:00:00+00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 2.5,
            },
            {
                "time": "1970-01-01T02:00:00+00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 2.5,
            },
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
