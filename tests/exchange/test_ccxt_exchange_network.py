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
