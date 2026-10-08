from datetime import UTC, datetime
from typing import Annotated

from loguru import logger
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fartt.candle_cache import CandleCache, interval_to_ms
from fartt.exchange import Candle, ExchangeUnavailable
from fartt.forecast import Forecaster, NotEnoughCandles, forecast

# Claude Code caps MCP tool output at about 25k tokens by default, and
# numbers cost roughly one token per three characters. 200 compact rows stay
# well under that, and the agent needs recent context, not years of history.
MAX_CANDLES = 200

COLUMNS = ["time", "open", "high", "low", "close", "volume"]

INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Every trading cycle starts with "
    "get_candles, which updates the cache from the exchange, then "
    "get_forecast for the next candle's expected return. If is_current is "
    "false the data is stale: read the warning and treat the cycle as a hold."
)

Row = tuple[str, float, float, float, float, float]


class CandlesResult(BaseModel):
    market: str
    interval: str
    columns: list[str] = Field(
        default=COLUMNS, description="Names of the values in each row, in order."
    )
    rows: list[Row] = Field(
        description=(
            "Closed candles, oldest first, one row per candle: time (open time, "
            "ISO 8601 in UTC), open, high, low, close, volume. The last row is "
            "the newest."
        )
    )
    new_candles: int = Field(
        description="Candles fetched from the exchange by this call."
    )
    is_current: bool = Field(
        description=(
            "True when the exchange was reached and the last row is the most "
            "recently closed period. False means the data may be stale; see "
            "warning."
        )
    )
    warning: str | None = Field(
        default=None, description="Why the data may be stale, if it may be."
    )


class ForecastResult(BaseModel):
    market: str
    interval: str
    model: str = Field(description="Name of the model that made the forecast.")
    expected_return: float = Field(
        description=(
            "Expected return of the next candle, as a fraction: 0.004 means "
            "+0.4%. The sign is the direction."
        )
    )
    applies_to: str = Field(
        description="Open time of the candle being forecast, ISO 8601 in UTC."
    )
    based_on: str = Field(
        description="Open time of the newest candle the forecast used, ISO 8601 in UTC."
    )
    is_current: bool = Field(
        description=(
            "True when the forecast is based on the most recently closed "
            "period. False means the cache is behind; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="Why the forecast may be stale, if it may be."
    )


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat()


def _row(candle: Candle) -> Row:
    time = _iso(candle.timestamp)
    return (time, candle.open, candle.high, candle.low, candle.close, candle.volume)


def build_server(
    cache: CandleCache, market: str, interval: str, forecaster: Forecaster
) -> MCPServer:
    """
    Build the MCP server for one market and interval, backed by `cache`.

    A factory rather than a module-level server, so tests can pass a cache
    over an in-memory exchange and connect with the SDK's in-memory client.

    """
    interval_ms = interval_to_ms(interval)
    server = MCPServer("fartt", instructions=INSTRUCTIONS)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
    def get_candles(
        count: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_CANDLES,
                description="How many of the latest closed candles to return.",
            ),
        ] = 50,
    ) -> CandlesResult:
        """
        Return the latest closed candles for the server's market and
        interval, oldest first, after bringing the local cache up to date
        from the exchange.
        """
        # The SDK reports any exception other than ToolError to the agent as
        # a bare "Error executing tool", so every failure the agent can't fix
        # by retrying is turned into a ToolError that says what to do.
        new_candles = 0
        warning: str | None = None
        try:
            for batch in cache.iter_update():
                new_candles += batch
        except ExchangeUnavailable as error:
            warning = (
                f"{error}. These are the cached candles, which may be stale: "
                "treat this cycle as a hold, or retry next cycle."
            )
        except ValueError as error:
            logger.error(f"get_candles: {error}")
            raise ToolError(
                f"{error}. The server is configured for {market} {interval}, "
                "which this exchange doesn't serve; retrying won't help. The "
                "operator must restart the server with a valid --market and "
                "--interval."
            ) from error
        except OSError as error:
            logger.error(f"get_candles: {error}")
            raise ToolError(
                f"Couldn't update the candle cache at {cache.filepath}: {error}. "
                "Retrying won't help until the operator fixes the file or its "
                "folder."
            ) from error

        candles = cache.latest(count)
        if not candles:
            raise ToolError(
                f"No {market} {interval} candles are cached yet"
                + (" and the exchange is unreachable" if warning else "")
                + ". Retry next cycle."
            )

        # A failed update means the newest candle couldn't be (re-)checked,
        # so the data is never "current" then, whatever the cache holds.
        is_current = (
            warning is None and candles[-1].timestamp == cache.newest_closed_ms()
        )
        if not is_current and warning is None:
            warning = (
                "The most recently closed candle is not published by the "
                "exchange yet; the last row here is the one before it. "
                "Retry shortly or treat this cycle as a hold."
            )

        logger.info(
            f"get_candles: {len(candles)} {market} {interval} candles, "
            f"{new_candles} new, current={is_current}"
        )
        return CandlesResult(
            market=market,
            interval=interval,
            rows=[_row(c) for c in candles],
            new_candles=new_candles,
            is_current=is_current,
            warning=warning,
        )

    @server.tool(
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False)
    )
    def get_forecast() -> ForecastResult:
        """
        Forecast the return of the next candle of the server's market and
        interval from the cached candles. Call get_candles first: this tool
        reads the cache and never contacts the exchange.
        """
        try:
            candles = cache.latest(forecaster.required_candles)
        except OSError as error:
            logger.error(f"get_forecast: {error}")
            raise ToolError(
                f"Couldn't read the candle cache at {cache.filepath}: {error}. "
                "Retrying won't help until the operator fixes the file or its "
                "folder."
            ) from error

        try:
            result = forecast(forecaster, candles, interval_ms)
        except NotEnoughCandles as error:
            raise ToolError(
                f"{forecaster.name} needs {error.needed} closed {market} "
                f"{interval} candles and the cache holds {error.got}. Call "
                "get_candles to update it, or have the operator fill it with "
                f"`uv run fartt download --market {market} --interval {interval}`; "
                "otherwise retry next cycle."
            ) from error

        is_current = result.based_on_ms == cache.newest_closed_ms()
        warning = None
        if not is_current:
            warning = (
                "The newest cached candle is older than the most recently "
                "closed period, so this forecast may be stale. Call get_candles "
                "first to update the cache; if it also reports is_current: "
                "false, treat this cycle as a hold."
            )

        logger.info(
            f"get_forecast: {result.model} expects {result.expected_return:+.4%} "
            f"for {market} {interval}, current={is_current}"
        )
        return ForecastResult(
            market=market,
            interval=interval,
            model=result.model,
            expected_return=result.expected_return,
            applies_to=_iso(result.applies_to_ms),
            based_on=_iso(result.based_on_ms),
            is_current=is_current,
            warning=warning,
        )

    return server
