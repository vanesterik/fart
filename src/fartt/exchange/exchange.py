from typing import Protocol

from fartt.exchange.candle import Candle


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

        A batch can hold fewer than `limit` candles even when more history
        exists: the exchange may cap the request (ccxt caps Bitvavo at
        1440), and dropping the forming candle shortens the last batch. To
        page through history, continue from the last returned timestamp
        plus one interval, never from `since_ms + limit * interval`. An
        empty list means nothing closed in that window -- a gap in the
        exchange's history and "nothing newer has closed yet" look the same.

        """
        ...
