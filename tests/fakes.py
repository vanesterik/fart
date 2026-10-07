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
