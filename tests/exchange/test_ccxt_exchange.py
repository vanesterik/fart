from typing import Any

import ccxt
import pytest

from fartt.exchange import Candle, CcxtExchange, ExchangeUnavailable

HOUR_MS = 3_600_000


class StubClient:
    """Stands in for a ccxt client: canned markets, timeframes and rows."""

    def __init__(
        self, rows: list[list[float]] | None = None, error: Exception | None = None
    ) -> None:
        self.timeframes = {"1m": "1m", "1h": "1h", "1d": "1d"}
        self.rows = rows or []
        self.error = error
        self.load_markets_calls = 0
        self.fetch_calls: list[dict[str, Any]] = []

    def load_markets(self) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        self.load_markets_calls += 1
        return {"BTC/EUR": {}, "ETH/EUR": {}}

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        if self.error is not None:
            raise self.error
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


def test_fetch_closed_candles_judges_closed_by_time_the_request_was_sent() -> None:
    # The response arrives after the 1h candle's period ended, but the
    # exchange answered before that: the candle was still forming then.
    now = [2 * HOUR_MS - 1]

    class SlowClient(StubClient):
        def fetch_ohlcv(
            self, symbol: str, timeframe: str, since: int, limit: int
        ) -> list[list[float]]:
            rows = super().fetch_ohlcv(symbol, timeframe, since, limit)
            now[0] = 2 * HOUR_MS + 1
            return rows

    exchange = CcxtExchange(
        client=SlowClient(rows=[_row(0), _row(HOUR_MS)]), clock=lambda: now[0]
    )

    candles = exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)

    assert [c.timestamp for c in candles] == [0]


def test_supported_intervals_lists_the_exchange_timeframes() -> None:
    exchange = _exchange(StubClient(), now_ms=0)

    assert exchange.supported_intervals() == ["1m", "1h", "1d"]


def test_network_error_while_loading_markets_becomes_exchange_unavailable() -> None:
    client = StubClient(error=ccxt.NetworkError("connection reset"))
    exchange = _exchange(client, now_ms=0)

    with pytest.raises(
        ExchangeUnavailable, match="bitvavo.*connection reset"
    ) as raised:
        exchange.has_market("BTC/EUR")
    assert isinstance(raised.value.__cause__, ccxt.NetworkError)


def test_exchange_error_while_fetching_becomes_exchange_unavailable() -> None:
    client = StubClient()
    exchange = _exchange(client, now_ms=10 * HOUR_MS)
    exchange.has_market("BTC/EUR")  # markets load fine
    client.error = ccxt.ExchangeError("maintenance")

    with pytest.raises(ExchangeUnavailable, match="maintenance"):
        exchange.fetch_closed_candles("BTC/EUR", "1h", since_ms=0, limit=10)
