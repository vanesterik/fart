from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from fartt.candle_cache import CandleCache
from fartt.forecast import RepeatLastReturn
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


def _server(cache: CandleCache) -> Any:
    return build_server(cache, "BTC/EUR", "1h", RepeatLastReturn())


async def _get_forecast(cache: CandleCache) -> Any:
    async with Client(_server(cache)) as client:
        return await client.call_tool("get_forecast", {})


async def _get_candles(cache: CandleCache, **arguments: Any) -> Any:
    async with Client(_server(cache)) as client:
        return await client.call_tool("get_candles", arguments)


@pytest.mark.anyio
async def test_get_candles_returns_latest_closed_candles(tmp_path: Path) -> None:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    result = await _get_candles(_cache(exchange, tmp_path), count=2)

    assert not result.is_error
    assert result.structured_content == {
        "market": "BTC/EUR",
        "interval": "1h",
        "columns": ["time", "open", "high", "low", "close", "volume"],
        "rows": [
            ["1970-01-01T01:00:00+00:00", 100.0, 101.0, 99.0, 100.0, 2.5],
            ["1970-01-01T02:00:00+00:00", 100.0, 101.0, 99.0, 100.0, 2.5],
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

    result = await _get_candles(_cache(exchange, tmp_path), count=200)

    assert len(result.structured_content["rows"]) == 3


@pytest.mark.anyio
async def test_get_candles_rejects_count_out_of_range(tmp_path: Path) -> None:
    cache = _cache(FakeExchange([], now_ms=0), tmp_path)

    for count in (0, 201):
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
    assert [row[0] for row in content["rows"]] == [
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
    assert len(content["rows"]) == 2
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

    async with Client(_server(cache)) as client:
        tools = (await client.list_tools()).tools

    assert [tool.name for tool in tools] == ["get_candles", "get_forecast"]
    assert tools[0].annotations is not None
    assert tools[0].annotations.read_only_hint is True
    assert tools[0].input_schema["properties"]["count"]["maximum"] == 200
    forecast_tool = tools[1]
    assert forecast_tool.annotations is not None
    assert forecast_tool.annotations.read_only_hint is True
    assert forecast_tool.annotations.open_world_hint is False
    assert forecast_tool.input_schema.get("properties", {}) == {}
    assert forecast_tool.output_schema is not None
    expected_return = forecast_tool.output_schema["properties"]["expected_return"]
    assert "fraction" in expected_return["description"]


@pytest.mark.anyio
async def test_get_candles_at_the_maximum_count_stays_compact(tmp_path: Path) -> None:
    # Claude Code caps MCP tool output (about 25k tokens by default); numbers
    # cost roughly one token per three characters, so stay well under that.
    exchange = FakeExchange([candle(h) for h in range(200)], now_ms=200 * HOUR_MS)

    result = await _get_candles(_cache(exchange, tmp_path, batch_limit=1000), count=200)

    assert len(result.structured_content["rows"]) == 200
    assert len(result.content[0].text) < 40_000


@pytest.mark.anyio
async def test_get_candles_is_not_current_when_the_update_failed(
    tmp_path: Path,
) -> None:
    # The cache already holds the newest closed period, but it couldn't be
    # re-checked: one signal, not "current" plus a warning to hold.
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    cache = _cache(exchange, tmp_path)
    cache.update()
    exchange.reachable = False

    content = (await _get_candles(cache, count=2)).structured_content

    assert content["is_current"] is False
    assert "unavailable" in content["warning"]


@pytest.mark.anyio
async def test_get_candles_explains_a_market_the_exchange_does_not_list(
    tmp_path: Path,
) -> None:
    # `fartt serve` starts without checking the market when the exchange is
    # down at startup; once it's back, a mistyped market must be explained.
    exchange = FakeExchange([], now_ms=3 * HOUR_MS, markets={"ETH/EUR"})

    result = await _get_candles(_cache(exchange, tmp_path))

    assert result.is_error
    text = result.content[0].text
    assert "BTC/EUR" in text
    assert "restart" in text.lower()


@pytest.mark.anyio
async def test_get_candles_explains_a_cache_it_cannot_write(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    exchange = FakeExchange([candle(0)], now_ms=2 * HOUR_MS)
    cache = CandleCache(
        exchange=exchange,
        assets_dir=blocked,
        market="BTC/EUR",
        interval="1h",
        history_start_ms=0,
        clock=lambda: exchange.now_ms,
    )

    result = await _get_candles(cache)

    assert result.is_error
    assert "cache" in result.content[0].text
    assert "not-a-directory" in result.content[0].text


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
    assert result.structured_content == {
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
