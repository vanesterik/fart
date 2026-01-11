from csv import DictReader
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from python_bitvavo_api.bitvavo import Bitvavo  # type: ignore
from tabulate import tabulate
from tqdm import tqdm

from fart.common.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.settings import Candle, Interval, Settings


class Downloader:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = Bitvavo(
            {
                "APIKEY": self._settings.api_key,
                "APISECRET": self._settings.api_secret,
            }
        )
        self._validate_settings()
        self._log_settings()

    def download(self) -> None:
        filepath = self._determine_filepath(self._settings)
        candle_data = self._load_cached_candle_data(filepath)
        start_timestamp = self._determine_start_timestamp(candle_data)
        timestamp_list = self._calculate_timestamp_list(
            start_timestamp, interval=self._settings.interval
        )

        for start, end in tqdm(timestamp_list, desc="Downloading"):
            candles: List[Candle] = self._client.candles(  # type: ignore
                self._settings.market,
                self._settings.interval.value,
                start=self._convert_timestamp(start),
                end=self._convert_timestamp(end),
            )
            candles = self._process_candles(candles)
            candle_data.extend(candles)
            # Save after each batch to avoid data loss
            self._save_candle_data(candle_data, filepath)

    def _validate_settings(self):
        data_dir = self._settings.data_dir
        market = self._settings.market
        markets = self._client.markets()  # type: ignore

        if not any(item["market"] == market for item in markets):
            raise ValueError(f"Market '{market}' not found in Bitvavo markets")

        # Ensure data directory exists, create it otherwise
        data_dir.mkdir(parents=True, exist_ok=True)

    def _log_settings(self):
        settings_ = self._settings.model_dump()
        settings_["api_secret"] = "*" * len(self._settings.api_key or "")
        settings_["interval"] = self._settings.interval.value
        table = tabulate(settings_.items())
        logger.info(f"\n\nFART // DOWNLOADER\n\n{table}\n")

    def _determine_filepath(self, settings: Settings) -> Path:
        data_dir = settings.data_dir
        market = settings.market
        interval = settings.interval.value
        return data_dir / f"{market}-{interval}.csv"

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
        # Return the timestamp of the last candle in the data, or the Bitvavo
        # launch timestamp if no data is available. The Bitvavo exchange
        # launched on March 9, 2019.
        bitvavo_launch_timestamp = 1552089600000  # 2019/03/09
        return data[-1][0] if data else bitvavo_launch_timestamp

    def _calculate_timestamp_list(
        self,
        start_timestamp: int,
        interval: Interval = Interval.ONE_DAY,
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
        interval: Interval = Interval.ONE_DAY,
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
