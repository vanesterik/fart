import json
from os import getenv
from typing import Any, Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo


def main() -> None:
    websocket = client.newWebsocket()
    websocket.setErrorCallback(errorCallback)
    websocket.subscriptionCandles("BTC-GUL", "1m", callback)

    # Keep the websocket alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Exiting...")


def callback(response: Any) -> None:
    print(response)


def errorCallback(error: Any) -> None:
    print("Error callback:", json.dumps(error, indent=2))


if __name__ == "__main__":

    # find and load .env files automagically
    load_dotenv(find_dotenv())

    # Create a Bitvavo client
    client = Bitvavo(
        {
            "APIKEY": getenv("BITVAVO_API_KEY"),
            "APISECRET": getenv("BITVAVO_API_SECRET"),
        }
    )

    main()
