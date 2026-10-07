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
