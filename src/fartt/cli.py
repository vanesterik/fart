import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from fartt.candle_cache import DEFAULT_HISTORY_START_MS, CandleCache
from fartt.exchange import CcxtExchange, ExchangeUnavailable
from fartt.server import build_server

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def _add_cache_options(command: argparse.ArgumentParser, interval: str) -> None:
    command.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="folder the candle cache lives in (default: %(default)s)",
    )
    command.add_argument(
        "--exchange", default="bitvavo", help="ccxt exchange id (default: %(default)s)"
    )
    command.add_argument(
        "--market",
        default="BTC/EUR",
        help="market in ccxt's format, e.g. BTC/EUR (default: %(default)s)",
    )
    command.add_argument(
        "--interval",
        default=interval,
        help="candle interval the exchange offers, e.g. 1m, 30m, 1h, 4h, 1d "
        "(default: %(default)s)",
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
    _add_cache_options(download, interval="1d")

    serve = commands.add_parser(
        "serve",
        help="run the MCP server over stdio (started by Claude Code from .mcp.json)",
        description="Run the MCP server for one market and interval over stdio.",
    )
    _add_cache_options(serve, interval="1h")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.add("logs/cli.log", rotation="1 MB", level="INFO", format=LOG_FORMAT)
    load_dotenv(find_dotenv())

    if args.command == "download":
        _download(parser, args.assets_dir, args.exchange, args.market, args.interval)
    elif args.command == "serve":
        _serve(parser, args.assets_dir, args.exchange, args.market, args.interval)


def _open_cache(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
    require_exchange: bool,
) -> CandleCache:
    # Validate the options before anything touches the cache, so a typo
    # exits with a one-line message rather than a traceback.
    try:
        client = CcxtExchange(exchange_id=exchange)
    except ValueError:
        parser.error(
            f"unknown exchange '{exchange}'; use a ccxt exchange id, e.g. 'bitvavo'"
        )

    supported = client.supported_intervals()
    if interval not in supported:
        parser.error(
            f"interval '{interval}' is not offered by exchange '{exchange}' "
            f"(supported: {', '.join(supported)})"
        )

    try:
        known = client.has_market(market)
    except ExchangeUnavailable as error:
        if require_exchange:
            parser.exit(1, f"fartt: error: {error}\n")
        logger.warning(f"{error}; starting anyway without checking market '{market}'")
        known = True
    if not known:
        parser.error(
            f"market '{market}' not found on exchange '{exchange}'. "
            "Markets use ccxt's format, e.g. 'BTC/EUR' rather than 'BTC-EUR'."
        )

    try:
        return CandleCache(
            exchange=client,
            assets_dir=assets_dir,
            market=market,
            interval=interval,
            history_start_ms=DEFAULT_HISTORY_START_MS,
        )
    except ValueError:
        # The exchange offers it, but the cache can't size it (e.g. a
        # lowercase `1w` or seconds-based `1s`); see issue #59.
        parser.error(
            f"interval '{interval}' is offered by exchange '{exchange}' but not "
            "supported by the candle cache (units: m, h, d, W, M)"
        )


def _download(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
) -> None:
    cache = _open_cache(
        parser, assets_dir, exchange, market, interval, require_exchange=True
    )
    configuration = {
        "exchange": exchange,
        "market": market,
        "interval": interval,
        "filepath": str(cache.filepath),
    }
    logger.info(f"\n\nFartt Downloader\n\n{tabulate(configuration.items())}\n")

    try:
        with tqdm(desc="Downloading", unit=" candles") as progress:
            for count in cache.iter_update():
                progress.update(count)
    except ExchangeUnavailable as error:
        parser.exit(1, f"fartt: error: {error}; the candles fetched so far are kept\n")


def _serve(
    parser: argparse.ArgumentParser,
    assets_dir: Path,
    exchange: str,
    market: str,
    interval: str,
) -> None:
    cache = _open_cache(
        parser, assets_dir, exchange, market, interval, require_exchange=False
    )
    logger.info(f"Serving {market} {interval} candles from {cache.filepath} over stdio")
    build_server(cache, market, interval).run()


if __name__ == "__main__":
    main()
