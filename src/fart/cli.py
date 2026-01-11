import sys
from os import getenv
from typing import Optional

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from fart.downloader import Downloader
from fart.settings import Settings
from fart.utils import update_settings

app = typer.Typer(no_args_is_help=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/cli.log", rotation="1 MB", level="INFO")

load_dotenv(find_dotenv())


@app.command()
def download(
    data_dir: Optional[str] = typer.Option(
        None,
        help="Folder to save downloaded data (defaults to system cache directory).",
    ),
    market: Optional[str] = typer.Option(
        None,
        help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC').",
    ),
    interval: Optional[str] = typer.Option(
        None,
        help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M').",
    ),
) -> None:
    arguments = {
        "data_dir": data_dir,
        "market": market,
        "interval": interval,
    }
    settings = Settings(
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
    )
    settings = update_settings(
        settings=settings,
        arguments=arguments,
    )

    downloader = Downloader(settings)
    downloader.download()


if __name__ == "__main__":
    app()
