from typing import NamedTuple


class Candle(NamedTuple):
    """
    One OHLCV candle. The field order matches the cached CSV columns
    (`Timestamp,Open,High,Low,Close,Volume`).

    A NamedTuple rather than a pydantic model: the 1m cache holds millions
    of rows, and per-row validation would make loading it slow.

    """

    timestamp: int  # Candle open time, milliseconds since the epoch (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float
