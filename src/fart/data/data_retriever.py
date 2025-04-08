import csv
from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import polars as pl
from loguru import logger
from python_bitvavo_api.bitvavo import Bitvavo

from fart.common.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME

# Define type alias for candle data
CandleData = Tuple[int, float, float, float, float, float]


class DataRetriever:
    def __init__(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        output_dir: str = "data",
    ) -> None:
        """
        Initialize the BitvavoData class containing methods to retrieve candle
        data from the Bitvavo API.

        Parameters
        ----------
        - api_key (str): API key for the Bitvavo API
        - api_secret (str): API secret for the Bitvavo API
        - output_dir (str): Output directory to save the data to

        """

        # Create a Bitvavo client
        self._client = Bitvavo(
            {
                "APIKEY": api_key,
                "APISECRET": api_secret,
            }
        )

        # Define output directory
        self._output_dir = Path(output_dir)

    def retrieve(
        self,
        market: str,
        interval: str,
    ) -> None:
        """
        Retrieve candle data from Bitvavo API and save it to a CSV file.

        By looping between start and end timestamps, this function retrieves
        candle data containing maximum amount of allowed entries (1400). The
        data is saved to a CSV file in the `/data` directory.

        Parameters
        ----------
        - market (str): Market to retrieve data for - i.e. "BTC-EUR".
        - interval (str): Interval of candle data cycle - i.e. "1m", "1h", etc.

        """

        if not self._is_valid_market(market):
            logger.error(f"Market {market} not found in Bitvavo API")
            return

        # Define output path based on market
        output_path = self._output_dir / f"{market}.csv"

        # Define variables in order to iterate over timestamp range
        data = self._get_or_create_data_list(output_path)
        start, end = self._determine_timestamps(data)
        timestamp_margin = self._calculate_time_margin(interval)

        # Loop over timestamps to retrieve candle data
        while start + timestamp_margin < end:
            start_timestamp = self._convert_timestamp(start)
            chunk = self._retrieve_candle_data(
                market,
                interval,
                start_timestamp,
            )
            data.extend(chunk)
            start += chunk[0][0] - chunk[-1][0]
            logger.info(f"Processing datetime: {start_timestamp}")

        # Add latest candles to data list that are left out of the loop. This to
        # make sure that the latest data is included in the data list
        data.extend(
            self._retrieve_candle_data(
                market,
                interval,
            )
        )
        logger.info(f"Adding latest candles to data list")

        # Create and save Polars DataFrame to output path
        self._create_and_save_dataframe(data, output_path)
        logger.success(f"Candle data saved to {output_path}")

    def _is_valid_market(self, market: str) -> bool:
        """
        Check if the passed market is valid.

        Parameters
        ----------
        - market (str): Market to check

        Returns
        -------
        - bool: True if the market is valid, False otherwise

        """
        return any(item["market"] == market for item in self._client.markets())

    def _get_or_create_data_list(self, output_path: Path) -> List[CandleData]:
        """
        Get the existing DataFrame or create a new one.

        Parameters
        ----------
        - output (Path): Path to the output file
        - schema (pl.Schema): Polars schema for the DataFrame

        Returns
        -------
        - pl.DataFrame: Existing or new DataFrame

        """

        data: List[Tuple[int, float, float, float, float, float]] = []

        # Return empty data list if the output path does not exist
        if not output_path.exists():
            return data

        # Otherwise, read CSV file and append raw data to the data list
        with open(output_path, "r", newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.DictReader(csvfile)
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

        # Return data list with processed data
        return data

    def _determine_timestamps(self, data: List[CandleData]) -> Tuple[int, int]:
        """
        Set the start and end timestamps for the candle data. If the DataFrame
        is empty, the start timestamp is set to the Bitvavo launch timestamp.
        Otherwise, the start timestamp is set to the last timestamp in the
        DataFrame.

        Parameters
        ----------
        - data (List[Tuple[int, float, float, float, float, float]]): Data list
          containing candle data

        Returns
        -------
        - Tuple[int, int]: Start and end timestamps

        """

        # Define Bitvavo launch timestamp, this to get all available data from
        # the start of Bitvavo
        bitvavo_launch_timestamp = 1552089600000  # 2019/03/09

        # Define start and end timestamps
        start = data[-1][0] if data else bitvavo_launch_timestamp
        end = int(time() // 60 * 60 * 1000)

        return start, end

    def _calculate_time_margin(self, interval: str) -> int:
        """
        Calculate the time margin based on the passed interval. This is used to
        ensure that the start timestamp is not set to a time that is too close
        to the current time, preventing the retrieval of duplicate data and
        getting stuck in a loop.

        Parameters
        ----------
        - interval (str): Interval of candle data cycle - i.e. "1m", "1h", etc.

        Returns
        -------
        - int: Time margin in milliseconds

        """

        if interval.endswith("m"):
            return int(interval[:-1]) * 60 * 1000
        if interval.endswith("h"):
            return int(interval[:-1]) * 60 * 60 * 1000
        if interval.endswith("d"):
            return int(interval[:-1]) * 24 * 60 * 60 * 1000
        return 0

    def _convert_timestamp(self, timestamp: int) -> datetime:
        """
        Convert timestamp to datetime. The timestamp is divided by 1000 to
        convert it to seconds. This is necessary because the Bitvavo API returns
        timestamps in milliseconds, but requires them in seconds for the
        `candles` method.

        Parameters
        ----------
        - timestamp (int): Timestamp to convert

        Returns
        -------
        - datetime: Converted timestamp

        """
        return datetime.fromtimestamp(timestamp / 1000)

    def _retrieve_candle_data(
        self,
        market: str,
        interval: str,
        start_timestamp: Optional[datetime] = None,
    ) -> List[CandleData]:
        """
        Retrieve candle data from the Bitvavo API.

        Parameters
        ----------
        - market (str): Market to retrieve data for - i.e. "BTC-EUR".
        - interval (str): Interval of candle data cycle - i.e. "1m", "1h", etc.
        - start_timestamp (Optional[datetime]): Start timestamp for the data
          retrieval

        Returns
        -------
        - List[CandleData]: Candle data retrieved from the Bitvavo API

        """
        data: List[CandleData] = []

        # Retrieve candle data from the Bitvavo API
        if start_timestamp:
            data = self._client.candles(
                market,
                interval,
                # Use end option to get data from start_timestamp - this seem to
                # be the only way to get data from a specific timestamp
                end=start_timestamp,
            )
        else:
            data = self._client.candles(
                market,
                interval,
            )
        # Return retrieved candle data typed as CandleData list
        return data

    def _create_and_save_dataframe(
        self,
        data: List[CandleData],
        output_path: Path,
    ) -> None:
        """
        Create a Polars DataFrame from the candle data and save it to a CSV file.

        Parameters
        ----------
        - data (List[CandleData]): Candle data retrieved from the Bitvavo API
        - output_path (Path): Path to output file

        """
        # Create Polars DataFrame from the candle data
        df = pl.DataFrame(
            data,
            orient="row",
            schema=[
                (TIMESTAMP, pl.Int64()),
                (OPEN, pl.Float64()),
                (HIGH, pl.Float64()),
                (LOW, pl.Float64()),
                (CLOSE, pl.Float64()),
                (VOLUME, pl.Float64()),
            ],
        )

        # Sort and deduplicate the DataFrame before saving it
        df = df.sort(TIMESTAMP).unique(subset=[TIMESTAMP], keep="first")

        # Save the Polars DataFrame to the output path
        df.write_csv(output_path)
