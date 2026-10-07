import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from fartt.candle_cache import DEFAULT_HISTORY_START_MS, CandleCache
from fartt.exchange import CcxtExchange

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fartt", description="Fartt: Financial Analysis Real Time Trading."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download",
        help="download closed candles into the local cache",
        description="Download closed candles into the local cache, resuming where it left off.",
    )
    download.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="folder the candle cache lives in (default: %(default)s)",
    )
    download.add_argument(
        "--exchange", default="bitvavo", help="ccxt exchange id (default: %(default)s)"
    )
    download.add_argument(
        "--market",
        default="BTC/EUR",
        help="market in ccxt's format, e.g. BTC/EUR (default: %(default)s)",
    )
    download.add_argument(
        "--interval",
        default="1d",
        help="candle interval the exchange offers, e.g. 1m, 30m, 1h, 4h, 1d (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.add("logs/cli.log", rotation="1 MB", level="INFO", format=LOG_FORMAT)
    load_dotenv(find_dotenv())

    if args.command == "download":
        _download(
            parser,
            assets_dir=args.assets_dir,
            exchange=args.exchange,
            market=args.market,
            interval=args.interval,
        )


def _download(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
) -> None:
    client = CcxtExchange(exchange_id=exchange)
    if not client.has_market(market):
        parser.error(
            f"market '{market}' not found on exchange '{exchange}'. "
            "Markets use ccxt's format, e.g. 'BTC/EUR' rather than 'BTC-EUR'."
        )

    cache = CandleCache(
        exchange=client,
        assets_dir=assets_dir,
        market=market,
        interval=interval,
        history_start_ms=DEFAULT_HISTORY_START_MS,
    )
    configuration = {
        "exchange": exchange,
        "market": market,
        "interval": interval,
        "filepath": str(cache.filepath),
    }
    logger.info(f"\n\nFartt Downloader\n\n{tabulate(configuration.items())}\n")

    with tqdm(desc="Downloading", unit=" candles") as progress:
        for count in cache.iter_update():
            progress.update(count)


if __name__ == "__main__":
    main()
