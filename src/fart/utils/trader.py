import json
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel
from python_bitvavo_api.bitvavo import Bitvavo


class BalanceAsset(BaseModel):
    symbol: str
    available: str
    inOrder: str


class Price(BaseModel):
    market: str
    price: str


class Trade(BaseModel):
    id: str
    orderId: str
    clientOrderId: str
    timestamp: int
    market: str
    side: str
    amount: str
    price: str
    taker: bool
    fee: str
    feeCurrency: str
    settled: bool


Candle = Tuple[int, str, str, str, str, str]


class CandlesSubscription(BaseModel):
    event: str
    market: str
    interval: str
    candle: List[Candle]


class Trader:
    """
    Trader class to handle trading operations with Bitvavo API.

    This class is a wrapper around the Bitvavo API to handle trading operations
    with the Bitvavo exchange. It provides a simple interface to retrieve
    balance data, candle data, price data, etc. It also allows to set the market
    and interval for which the data should be retrieved.

    The Bitvavo API is not typed properly, so this class provides a typed API to
    interact with the Bitvavo API.

    Parameters
    ----------
    - api_key (str): The API key for the Bitvavo API.
    - api_secret (str): The API secret for the Bitvavo API.

    """

    def __init__(self, api_key: Optional[str], api_secret: Optional[str]) -> None:
        self._client = Bitvavo(
            {
                "APIKEY": api_key,
                "APISECRET": api_secret,
            }
        )

    @property
    def market(self) -> str:

        return self._market

    @market.setter
    def market(self, market: str) -> None:

        self._market = market

    @property
    def interval(self) -> str:

        return self._interval

    @interval.setter
    def interval(self, interval: str) -> None:

        self._interval = interval

    @property
    def balance(self) -> List[BalanceAsset]:

        balance_data = self._client.balance()
        return [BalanceAsset(**asset) for asset in balance_data]

    @property
    def last_candle(self) -> Candle:

        return self._get_candles(limit=1)[0]

    @property
    def price(self) -> Price:

        price_date = self._client.tickerPrice(options={"market": self._market})
        return Price(**price_date)

    @property
    def trades(self) -> List[Trade]:

        trades = self._client.trades(self._market)
        return [Trade(**trade) for trade in trades]

    def _get_candles(
        self,
        limit: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Candle]:

        candles: List[Candle] = self._client.candles(
            self._market,
            self._interval,
            None,
            limit,
            start,
            end,
        )

        return candles

    def initiate(self, callback: Callable[[Dict[str, Any]], None]) -> None:

        self._socket = self._client.newWebsocket()
        self._socket.subscriptionCandles(self._market, self._interval, callback)
        self._socket.setErrorCallback(self._error_callback)

    def _error_callback(self, error: Any) -> None:

        print("Errors:", json.dumps(error, indent=2))

    def wait_and_close(self) -> None:

        # Bitvavo uses a weight-based rate limiting system. The application is
        # limited to 1000 weight points per IP or API key per minute. The rate
        # weighting for each endpoint is provided in the Bitvavo API
        # documentation. This call returns the number of points left. If more
        # requests are made than permitted by the weight limit, the IP or API
        # key will be banned.
        limit = self._client.getRemainingLimit()
        try:
            while limit > 0:
                sleep(0.5)
                limit = self._client.getRemainingLimit()
        except KeyboardInterrupt:
            self._socket.closeSocket()
