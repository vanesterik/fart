from csv import DictReader
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from python_bitvavo_api.bitvavo import Bitvavo
from tabulate import tabulate
from tqdm import tqdm

from fart.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.utils import get_candle_filepath

Candle = Tuple[int, float, float, float, float, float]


class Downloader:
    def __init__(
        self,
        data_dir: Path,
        market: str,
        interval: str,
        api_key: str | None,
        api_secret: str | None,
    ):
        self._data_dir = data_dir
        self._market = market
        self._interval = interval
        self._client = Bitvavo(
            {
                "APIKEY": api_key,
                "APISECRET": api_secret,
            }
        )
        self._validate_market()
        self._determine_filepath()
        self._log_configuration()

    def download(self) -> None:
        filepath = self._filepath
        candle_data = self._load_cached_candle_data(filepath)
        start_timestamp = self._determine_start_timestamp(candle_data)
        timestamp_list = self._calculate_timestamp_list(
            start_timestamp, interval=self._interval
        )

        for start, end in tqdm(timestamp_list, desc="Downloading"):
            candles: List[Candle] = self._client.candles(
                self._market,
                self._interval,
                start=self._convert_timestamp(start),
                end=self._convert_timestamp(end),
            )
            candles = self._process_candles(candles)
            candle_data.extend(candles)
            # Save after each batch to avoid data loss
            self._save_candle_data(candle_data, filepath)

    def _validate_market(self):
        markets = self._client.markets()

        if not any(item["market"] == self._market for item in markets):
            raise ValueError(f"Market '{self._market}' not found in Bitvavo markets")

    def _determine_filepath(self):
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = get_candle_filepath(
            self._data_dir,
            self._market,
            self._interval,
        )

    def _log_configuration(self):
        configuration = {
            "data_dir": str(self._data_dir),
            "market": self._market,
            "interval": self._interval,
            "filepath": str(self._filepath),
        }
        table = tabulate(configuration.items())
        logger.info(f"\n\nF.A.R.T. Downloader\n\n{table}\n")

    def _load_cached_candle_data(self, filepath: Path) -> List[Candle]:
        if not filepath.exists():
            return []

        with open(filepath, "r", newline="", encoding="utf-8") as file:
            csv_reader = DictReader(file)
            data: List[Candle] = []

            for row in csv_reader:
                data.append(
                    (
                        int(row[TIMESTAMP]),
                        float(row[OPEN]),
                        float(row[HIGH]),
                        float(row[LOW]),
                        float(row[CLOSE]),
                        float(row[VOLUME]),
                    )
                )
            return data

    def _determine_start_timestamp(self, data: List[Candle]) -> int:
        # Return the timestamp one interval past the last candle in the
        # data (so the already-cached candle isn't re-fetched and appended
        # as a duplicate), or the Bitvavo launch timestamp if no data is
        # available. The Bitvavo exchange launched on March 9, 2019.
        bitvavo_launch_timestamp = 1552089600000  # 2019/03/09
        if not data:
            return bitvavo_launch_timestamp
        return data[-1][0] + self._interval_to_milliseconds(self._interval)

    def _interval_to_milliseconds(self, interval: str) -> int:
        if interval.endswith("m"):
            return int(interval[:-1]) * 60_000
        elif interval.endswith("h"):
            return int(interval[:-1]) * 3_600_000
        elif interval.endswith("d"):
            return int(interval[:-1]) * 86_400_000
        elif interval.endswith("W"):
            return int(interval[:-1]) * 604_800_000
        elif interval.endswith("M"):
            return int(interval[:-1]) * 30 * 86_400_000
        else:
            raise ValueError(f"Invalid interval: {interval}")

    def _calculate_timestamp_list(
        self,
        start_timestamp: int,
        interval: str = "1d",
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> List[Tuple[int, int]]:
        timestamps = [start_timestamp]
        end_timestamp = int(datetime.now().timestamp() * 1000)

        while start_timestamp < end_timestamp:
            next_timestamp = self._calculate_timestamp(
                timestamp=start_timestamp,
                epochs=epochs,
                interval=interval,
            )
            timestamps.append(next_timestamp)
            start_timestamp = next_timestamp

        return list(zip(timestamps, timestamps[1:]))

    def _calculate_timestamp(
        self,
        timestamp: int,
        interval: str = "1d",
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> int:
        # Convert milliseconds to seconds, then to datetime
        dt_ = datetime.fromtimestamp(timestamp / 1000)

        # Calculate time delta based on interval
        if interval.endswith("m"):
            delta = timedelta(minutes=int(interval[:-1]) * epochs)
        elif interval.endswith("h"):
            delta = timedelta(hours=int(interval[:-1]) * epochs)
        elif interval.endswith("d"):
            delta = timedelta(days=int(interval[:-1]) * epochs)
        elif interval.endswith("W"):
            delta = timedelta(weeks=int(interval[:-1]) * epochs)
        elif interval.endswith("M"):
            delta = timedelta(days=30 * int(interval[:-1]) * epochs)
        else:
            raise ValueError(f"Invalid interval: {interval}")

        # Add epochs
        dt = min(dt_ + delta, datetime.now())

        # Convert back to milliseconds
        return int(dt.timestamp() * 1000)

    def _convert_timestamp(self, timestamp: int) -> datetime:
        # Convert timestamp to datetime. The timestamp is divided by 1000 to
        # convert it to seconds. This is necessary because the Bitvavo API returns
        # timestamps in milliseconds, but requires them in seconds for the
        # `candles` method.
        return datetime.fromtimestamp(timestamp / 1000)

    def _process_candles(self, candles: List[Candle]) -> List[Candle]:
        return sorted(candles, key=lambda candle: candle[0])

    def _save_candle_data(self, candle_data: List[Candle], filepath: Path) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as file:
            file.write(f"{TIMESTAMP},{OPEN},{HIGH},{LOW},{CLOSE},{VOLUME}\n")
            for candle in candle_data:
                file.write(
                    f"{candle[0]},{candle[1]},{candle[2]},{candle[3]},{candle[4]},{candle[5]}\n"
                )
