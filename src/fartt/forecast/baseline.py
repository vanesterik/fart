from collections.abc import Sequence
from dataclasses import dataclass

from fartt.exchange import Candle


@dataclass(frozen=True)
class RepeatLastReturn:
    """
    Naive baseline: forecast that the next candle repeats the last candle's
    return. One of the PRD's naive baselines -- every model epic B selects
    has to beat it. Chosen over "no change" because it gives the trading
    loop realistic, non-zero signals to act on.

    """

    name: str = "repeat-last-return"
    kind: str = "naive baseline"
    required_candles: int = 2
    trained_through_ms: int | None = None

    def predict(self, candles: Sequence[Candle]) -> float:
        previous, last = candles[-2], candles[-1]
        return last.close / previous.close - 1
