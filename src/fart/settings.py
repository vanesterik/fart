from enum import Enum
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel

Candle = Tuple[int, float, float, float, float, float]


class Interval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"


class Settings(BaseModel):
    api_key: str | None = None
    api_secret: str | None = None
    data_dir: Path = Path.home() / ".cache/fart"
    market: str = "BTC-EUR"
    interval: Interval = Interval.ONE_DAY
