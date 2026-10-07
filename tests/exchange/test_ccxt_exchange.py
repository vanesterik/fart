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
