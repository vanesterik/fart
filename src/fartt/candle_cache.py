import time
from collections import deque
from collections.abc import Callable, Iterator
from csv import DictReader
from pathlib import Path

from fartt.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fartt.exchange import Candle, Exchange
from fartt.utils import get_data_filepath

# Bitvavo launched on 2019-03-09; nothing older exists there to fetch.
DEFAULT_HISTORY_START_MS = 1_552_089_600_000
DEFAULT_BATCH_LIMIT = 1000

HEADER = f"{TIMESTAMP},{OPEN},{HIGH},{LOW},{CLOSE},{VOLUME}\n"

# Enough bytes to hold the last line of any cache file (a line is ~60 bytes).
_TAIL_BYTES = 1024

_UNIT_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "W": 604_800_000,
    "M": 30 * 86_400_000,
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def interval_to_ms(interval: str) -> int:
    """Length of a candle interval such as `1h` or `30m`, in milliseconds."""
    unit = interval[-1:]
    if unit not in _UNIT_MS or not interval[:-1].isdigit():
        raise ValueError(f"Invalid interval: {interval}")
    return int(interval[:-1]) * _UNIT_MS[unit]


class CandleCache:
    """
    One market and interval's candles, cached as a CSV file under
    `assets_dir` and kept up to date through an `Exchange`.

    The file only ever grows by appending, with one exception: each update
    re-fetches the last cached candle, and if the exchange has amended it
    since, the file's last line is replaced. Both `fartt download` and the
    MCP server read and update the cache through this class.

    `clock` returns the current time in milliseconds and decides where
    "the present" is: paging stops after the newest closed period, so an
    empty batch before that point is a gap in the exchange's history and
    is skipped.

    """

    def __init__(
        self,
        exchange: Exchange,
        assets_dir: Path,
        market: str,
        interval: str,
        history_start_ms: int = DEFAULT_HISTORY_START_MS,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        self.filepath = get_data_filepath(assets_dir, market, interval)
        self._exchange = exchange
        self._assets_dir = assets_dir
        self._market = market
        self._interval = interval
        self._interval_ms = interval_to_ms(interval)
        self._history_start_ms = history_start_ms
        self._batch_limit = batch_limit
        self._clock = clock

    def iter_update(self) -> Iterator[int]:
        """
        Fetch every closed candle newer than the cache, one batch at a time,
        and yield the number of new candles after appending each batch.

        """
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        last = self._read_last_candle()
        since_ms = last.timestamp if last is not None else self._history_start_ms
        now_ms = self._clock()
        newest_closed_ms = (now_ms // self._interval_ms - 1) * self._interval_ms

        while since_ms <= newest_closed_ms:
            batch = self._exchange.fetch_closed_candles(
                self._market, self._interval, since_ms, self._batch_limit
            )
            if not batch:
                # Nothing closed in this window although the present is
                # further on: a gap in the exchange's history. Skip it.
                since_ms += self._batch_limit * self._interval_ms
                continue

            if last is not None and batch[0].timestamp == last.timestamp:
                if batch[0] != last:
                    self._replace_last_line(batch[0])
                    last = batch[0]

            new = [c for c in batch if last is None or c.timestamp > last.timestamp]
            if new:
                self._append(new)
                last = new[-1]
                yield len(new)

            since_ms = batch[-1].timestamp + self._interval_ms

    def update(self) -> int:
        """Run `iter_update()` to the end; return the number of new candles."""
        return sum(self.iter_update())

    def latest(self, count: int) -> list[Candle]:
        """Return up to `count` of the newest cached candles, oldest first."""
        if not self.filepath.exists():
            return []
        with self.filepath.open(newline="", encoding="utf-8") as file:
            rows = deque(DictReader(file), maxlen=count)
        return [
            Candle(
                timestamp=int(row[TIMESTAMP]),
                open=float(row[OPEN]),
                high=float(row[HIGH]),
                low=float(row[LOW]),
                close=float(row[CLOSE]),
                volume=float(row[VOLUME]),
            )
            for row in rows
        ]

    def _read_tail(self) -> tuple[int, bytes]:
        # Returns (offset of the tail in the file, the tail's bytes), reading
        # at most _TAIL_BYTES so a multi-million-row cache resumes instantly.
        with open(self.filepath, "rb") as file:
            file.seek(0, 2)
            size = file.tell()
            offset = max(0, size - _TAIL_BYTES)
            file.seek(offset)
            return offset, file.read()

    def _read_last_candle(self) -> Candle | None:
        if not self.filepath.exists():
            return None
        _, tail = self._read_tail()
        last_line = tail.rstrip(b"\n").rsplit(b"\n", 1)[-1].decode("utf-8")
        if not last_line or last_line.startswith(TIMESTAMP):
            return None
        values = last_line.split(",")
        return Candle(int(values[0]), *(float(v) for v in values[1:6]))

    def _replace_last_line(self, candle: Candle) -> None:
        offset, tail = self._read_tail()
        start = offset + tail.rstrip(b"\n").rfind(b"\n") + 1
        with open(self.filepath, "rb+") as file:
            file.truncate(start)
        self._append([candle])

    def _append(self, candles: list[Candle]) -> None:
        is_new = not self.filepath.exists() or self.filepath.stat().st_size == 0
        with self.filepath.open("a", newline="", encoding="utf-8") as file:
            if is_new:
                file.write(HEADER)
            file.writelines(
                ",".join(str(value) for value in candle) + "\n" for candle in candles
            )
