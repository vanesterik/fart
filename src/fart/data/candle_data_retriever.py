import csv
import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from python_bitvavo_api.bitvavo import Bitvavo

from fart.common.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME

# Define type alias for candle data
CandleData = Tuple[int, float, float, float, float, float]


class CandleDataRetriever:
    def __init__(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        output_dir: str = "data/raw",
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
        symbol: str,
        interval: str,
    ) -> None:
        """
        Retrieve candle data from Bitvavo API and save it to a CSV file.

        By looping between start and end timestamps, this function retrieves
        candle data containing maximum amount of allowed entries (1400). The
        data is saved to a CSV file in the `/data` directory.

        Parameters
        ----------
        - symbol : str
            Symbol to retrieve data for - i.e. "BTC-EUR".

        - interval : str
            Interval of candle data cycle - i.e. "1m", "1h", etc.

        """

        if not self._is_valid_symbol(symbol):
            logger.error(f"Symbol {symbol} not found in Bitvavo API")
            return

        # Define output path based on market
        output_path = self._output_dir / f"{symbol}-{interval}.csv"

        # Create or get existing data file
        data = self._create_or_get_data_file(output_path)

        # Determine interval duration in milliseconds
        interval_duration = self._determine_interval_duration(interval)

        # Determine start and end timestamps for the candle data
        start, end = self._determine_boundary_timestamps(interval_duration, data)

        # Loop over timestamps to retrieve candle data
        while start + interval_duration < end:
            # Convert timestamps to datetime objects
            start_timestamp = self._convert_timestamp(start)
            end_timestamp = self._convert_timestamp(end)
            # Retrieve candle data from Bitvavo API
            data = self._retrieve_candle_data(
                symbol=symbol,
                interval=interval,
                start=start_timestamp,
                end=end_timestamp,
            )
            # Update start and end timestamps for the next iteration
            start, end = self._determine_boundary_timestamps(interval_duration, data)
            # Save the retrieved candle data to the output file
            self._save_candle_data(data, output_path)

            logger.info(f"{start_timestamp} - {end_timestamp}")

        logger.success(f"Candle data saved to {output_path}")

    def _is_valid_symbol(self, symbol: str) -> bool:
        """
        Check if the passed symbol is valid.

        Parameters
        ----------
        - symbol : str
            Symbol to check

        Returns
        -------
        - boolean : bool
            True if the symbol is valid, False otherwise

        """
        return any(item["market"] == symbol for item in self._client.markets())

    def _create_or_get_data_file(self, output_path: Path) -> List[CandleData]:
        """
        Create or get existing data file.

        If the file does not exist, it will create a new one and return an empty
        list. If the file exists, it will read the data from the file and return
        it as a list of tuples.

        Parameters
        ----------
        - output : Path
            Path to the output file

        Returns
        -------
        - data : List[CandleData]
            List of candle data tuples

        """

        # Define data list to store candle data
        data: List[CandleData] = []

        # Return empty data list if the output path does not exist
        if not output_path.exists():
            with open(
                encoding="utf-8",
                file=output_path,
                mode="w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        TIMESTAMP,
                        OPEN,
                        HIGH,
                        LOW,
                        CLOSE,
                        VOLUME,
                    ]
                )
            # Return empty data list as no data is available yet
            return data

        # Otherwise, read CSV file and append raw data to the data list
        with open(
            encoding="utf-8",
            file=output_path,
            mode="r",
            newline="",
        ) as file:
            dict_reader = csv.DictReader(file)
            for row in dict_reader:
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
        # Return data list from output file
        return data

    def _determine_boundary_timestamps(
        self, interval_duration: int, data: List[CandleData]
    ) -> Tuple[int, int]:
        """
        Set the start and end timestamps for the candle data. If the DataFrame
        is empty, the start timestamp is set to the Bitvavo launch timestamp.
        Otherwise, the start timestamp is set to the last timestamp in the
        DataFrame.

        Parameters
        ----------
        - interval_duration : int
            Duration of candle data interval in milliseconds

        - data : List[CandleData]
            Data list containing candle data

        Returns
        -------
        - start, end : Tuple[int, int]
            Start and end timestamps

        """

        # Define Bitvavo launch timestamp, this to get all available data from
        # the start of Bitvavo
        bitvavo_launch_timestamp = 1552089600000  # 2019/03/09
        request_limit = 1400  # Maximum amount of entries per request

        # Define start and end timestamps
        start = data[-1][0] if data else bitvavo_launch_timestamp
        end = start + interval_duration * request_limit
        # Define now timestamp
        now = int(dt.datetime.now().timestamp() * 1000)
        # Ensure end timestamp does not exceed current time
        end = min(end, now)

        return start, end

    def _determine_interval_duration(self, interval: str) -> int:
        """
        Determine the millisecond interval based on the passed interval. This is
        used to ensure that the start timestamp is set to a time that is not too
        close to the current time, preventing the retrieval of duplicate data
        and getting stuck in a loop.

        Parameters
        ----------
        - interval : str
            Interval of candle data cycle - i.e. "1m", "1h", etc.

        Returns
        -------
        - int: Millisecond interval

        """
        # Define the mapping of intervals to milliseconds
        interval_mapping = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }

        return interval_mapping.get(interval, 0)

    def _convert_timestamp(self, timestamp: int) -> dt.datetime:
        """
        Convert timestamp to datetime. The timestamp is divided by 1000 to
        convert it to seconds. This is necessary because the Bitvavo API returns
        timestamps in milliseconds, but requires them in seconds for the
        `candles` method.

        Parameters
        ----------
        - timestamp : int
            Timestamp to convert

        Returns
        -------
        - timestamp : dt.datetime
            Converted timestamp

        """
        return dt.datetime.fromtimestamp(timestamp / 1000)

    def _retrieve_candle_data(
        self,
        symbol: str,
        interval: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> List[CandleData]:
        """
        Retrieve candle data from the Bitvavo API.

        Parameters
        ----------
        - symbol : str
            Market to retrieve data for - i.e. "BTC-EUR".

        - interval : str
            Interval of candle data cycle - i.e. "1m", "1h", etc.

        - start : dt.datetime
            Start timestamp for the data retrieval

        - end : dt.datetime
            End timestamp for the data retrieval

        Returns
        -------
        - data : List[CandleData]
            Candle data retrieved from the Bitvavo API

        """
        data: List[CandleData] = self._client.candles(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )

        # Revert the data to ensure it is in the correct order
        data.reverse()

        # Return retrieved candle data typed as CandleData list
        return data

    def _save_candle_data(self, data: List[CandleData], output_path: Path) -> None:
        """
        Save a chunk of candle data to the output file.

        Parameters
        ----------
        - data : List[CandleData]
            List of candle data tuples to save

        - output_path : Path
            Path to the output file

        """
        if not data:
            return

        # Create or get existing data file
        with open(
            encoding="utf-8",
            file=output_path,
            mode="a",
            newline="",
        ) as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)
