from datetime import UTC, datetime
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from fartt.candle_cache import CandleCache
from fartt.exchange import Candle, ExchangeUnavailable

MAX_CANDLES = 1000

INSTRUCTIONS = (
    "Fartt's trading server. It serves one market and one candle interval, "
    "fixed when the server starts. Start every trading cycle with "
    "get_candles. If is_current is false the data is stale: read the warning "
    "and treat the cycle as a hold."
)


class CandleOut(BaseModel):
    time: str = Field(description="Candle open time, ISO 8601 in UTC.")
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlesResult(BaseModel):
    market: str
    interval: str
    candles: list[CandleOut] = Field(
        description="Closed candles, oldest first; the last is the newest."
    )
    new_candles: int = Field(
        description="Candles fetched from the exchange by this call."
    )
    is_current: bool = Field(
        description=(
            "True when the last candle is the most recently closed period. "
            "False means the data is stale; see warning."
        )
    )
    warning: str | None = Field(
        default=None, description="Why the data may be stale, if it may be."
    )


def _candle_out(candle: Candle) -> CandleOut:
    return CandleOut(
        time=datetime.fromtimestamp(candle.timestamp / 1000, UTC).isoformat(),
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


def build_server(cache: CandleCache, market: str, interval: str) -> MCPServer:
    """
    Build the MCP server for one market and interval, backed by `cache`.

    A factory rather than a module-level server, so tests can pass a cache
    over an in-memory exchange and connect with the SDK's in-memory client.

    """
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

        candles = cache.latest(count)
        if not candles:
            raise ToolError(
                f"No {market} {interval} candles are cached yet"
                + (" and the exchange is unreachable" if warning else "")
                + ". Retry next cycle."
            )

        is_current = candles[-1].timestamp == cache.newest_closed_ms()
        if not is_current and warning is None:
            warning = (
                "The most recently closed candle is not published by the "
                "exchange yet; the last candle here is the one before it. "
                "Retry shortly or treat this cycle as a hold."
            )

        return CandlesResult(
            market=market,
            interval=interval,
            candles=[_candle_out(c) for c in candles],
            new_candles=new_candles,
            is_current=is_current,
            warning=warning,
        )

    return server
