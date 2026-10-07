# Server Foundation — Design

**Date:** 2026-10-07
**Status:** Approved
**Source:** [Epic #44: Server Foundation](https://github.com/vanesterik/fartt/issues/44), stories [#50](https://github.com/vanesterik/fartt/issues/50), [#51](https://github.com/vanesterik/fartt/issues/51), [#52](https://github.com/vanesterik/fartt/issues/52) and [#53](https://github.com/vanesterik/fartt/issues/53)
**Related PRD:** `docs/product/mcp-trading-agent-prd.md` (§5 Solution Overview, §7 epic A)

## Problem

The PRD replaces the self-built trading platform with an MCP server that Claude Code operates. Epic A lays the foundation every later tool builds on:

- **Exchange access** goes through an internal interface backed by ccxt, so Bitvavo is not a lock-in.
- **The candle cache** serves both offline training (`fartt download`) and the agent.
- **The MCP server** exposes the first tool, `get_candles`.
- **Superseded code goes:** the unwired Part B scaffolding and the second Bitvavo integration.

Today `downloader.py` calls `python_bitvavo_api` directly. `core/exchange.py` is a second, unused Bitvavo wrapper, and `core/dashboard.py` and `core/broker.py` are unwired scaffolding.

## Findings that shape the design

Checked on 2026-10-07 against ccxt 4.5.85 and the MCP Python SDK v2.3:

- **Candles:** ccxt's Bitvavo `fetch_ohlcv` supports `1m` through `1d`. Its last row is the still-forming candle.
- **No sandbox:** ccxt lists no test environment for Bitvavo, so paper mode has to be simulated by the server (epic D).
- **Stop-losses:** `create_order` with a `stopLossPrice` parameter maps to Bitvavo's native `stopLoss`/`stopLossLimit` order types. The stop is a separate order placed after the entry fills (epic F).
- **`operatorId`:** Bitvavo requires one on every order, or ccxt refuses `create_order`. It becomes a config value in epic F.
- **Fees:** ccxt's fee table puts Bitvavo's entry tier (under €100k 30-day volume) at 0.25% taker and 0.15% maker. That is a 0.5% round trip with taker orders, which confirms the PRD's assumption. The operator's actual tier still needs an authenticated check.
- **MCP SDK v2:** servers are `mcp.server.MCPServer`. Tools are decorated functions, a Pydantic return type becomes structured output, errors are raised as `ToolError`, and `run()` defaults to stdio, where stdout carries the protocol.

## Decisions

- **Market symbols use ccxt's format, `BTC/EUR`,** in tool arguments and the CLI. Cache file names keep `BTC-EUR`: `get_data_filepath` converts `/` to `-`, so existing files and notebook calls that pass `BTC-EUR` resolve to the same path.
- **Server settings are CLI options on a new `fartt serve` command,** set in `.mcp.json`. A config file arrives in epic D, when the risk limits need one.
- **The server trades one configured market and interval.** `get_candles` takes no market or interval argument.
- **The downloader is split into a reusable cache** (approach A). `fartt download` and `get_candles` share one cache path.
- **No API keys in epic A.** Candles are public, and keys come back in epic D or F under names that don't mention Bitvavo.

## Design

### 1. Exchange layer (story #50)

```
src/fartt/exchange/
├── __init__.py
├── candle.py          # Candle NamedTuple
├── exchange.py        # Exchange Protocol
└── ccxt_exchange.py   # CcxtExchange: the only module that imports ccxt
```

- **`Candle`** is a `NamedTuple` with the fields `timestamp` (ms), `open`, `high`, `low`, `close` and `volume`. That's the same order as today's tuple and the CSV columns. It's light enough for millions of 1m rows, where a Pydantic model per row would be slow.
- **`Exchange`** is a `Protocol`, like `ModelBuilder`, with only what epic A needs:
  - `has_market(market: str) -> bool`
  - `fetch_closed_candles(market: str, interval: str, since_ms: int, limit: int) -> list[Candle]`

  Balances and orders get added in epics D and F, when they have callers.
- **`CcxtExchange(exchange_id: str = "bitvavo", client=None, clock=None)`:**
  - Builds the ccxt client from `exchange_id` with the rate limiter on, and loads markets once.
  - `fetch_closed_candles` calls `fetch_ohlcv`, drops every candle whose period hasn't ended by `clock()` (so the forming candle never leaves the layer), and returns candles sorted by timestamp.
  - `client` and `clock` are injectable for tests.
- **Errors:**
  - An unknown market raises `ValueError` naming the market and the exchange.
  - ccxt's network and exchange errors pass through unchanged. The cache and the server decide how to present them.

### 2. Candle cache and `fartt download` (story #51)

`src/fartt/candle_cache.py` holds `CandleCache(exchange, assets_dir, market, interval, history_start_ms=<2019-03-09>)`:

- **`iter_update() -> Iterator[int]`** fetches every closed candle newer than the cache, one batch at a time. After **appending** each batch to the CSV, it yields the batch size.
  - It resumes at the last cached timestamp plus one interval, or at `history_start_ms` when the cache is empty.
  - When a batch comes back empty before reaching the present (a gap in the exchange's history), it skips ahead one batch window instead of stopping.
  - A timestamp already in the cache is never written again.
  - Appending replaces today's full rewrite of the CSV after every batch.
- **`update() -> int`** runs `iter_update()` to the end and returns the number of new candles.
- **`latest(count: int) -> list[Candle]`** returns the last `count` cached candles.
- The interval-to-milliseconds helper moves here from `Downloader`.

**`fartt download`** builds `CcxtExchange` and `CandleCache` and wraps `iter_update()` in tqdm. Its options are `--exchange` (default `bitvavo`), `--market` (default `BTC/EUR`), `--interval` and `--assets-dir`, and the tabulate banner stays. `downloader.py` and the `Downloader` class are deleted.

### 3. MCP server and `get_candles` (story #52)

```
src/fartt/server/
├── __init__.py
└── server.py      # build_server(cache, market, interval) -> MCPServer
```

- **`build_server(...)`** is a factory, so tests can pass a cache backed by a fake exchange. The server's `instructions` tell the agent that it serves one configured market and interval and that every cycle starts with `get_candles`.
- **`get_candles(count: int = 50)`** accepts 1 to 1000. It calls `cache.update()` and returns a Pydantic model:
  - `market`, `interval`
  - `candles`: a list of `{time (ISO 8601 UTC), open, high, low, close, volume}`
  - `new_candles`: how many were fetched by this call
  - `is_current`: whether the last candle is the most recently closed period
  - `warning`: `null` unless something degraded
- **Errors, written for the agent:**
  - **Exchange unreachable, cache has data:** return the cached candles with `is_current: false` and a `warning` explaining why, and let the agent decide whether stale data means "hold".
  - **Exchange unreachable, cache empty:** `ToolError` saying no data is available and suggesting a retry next cycle.
  - **`count` out of range:** rejected with the allowed range.
- **`fartt serve`** takes `--exchange`, `--market`, `--interval` and `--assets-dir`. It builds the exchange and the cache, checks `has_market` at startup (exiting with a clear message if the check fails), and runs the server over stdio. Logging stays on loguru's existing stderr and `logs/cli.log` sinks, and nothing writes to stdout.
- **`.mcp.json`** registers the server for the project:
  ```json
  {"mcpServers": {"fartt": {"command": "uv", "args": ["run", "fartt", "serve", "--market", "BTC/EUR", "--interval", "1h"]}}}
  ```
  `1h` is a placeholder until epic B chooses the interval.
- **Dependencies added:** `mcp` (v2) and `ccxt`.
- **A known one-off cost:** with an empty cache, the first call fills the history from 2019 (about 60 batches at 1h). The README says to run `fartt download` first.

### 4. Removal and documentation (story #53)

- **Removed:**
  - `src/fartt/core/` (`exchange.py`, `dashboard.py`, `broker.py`). Nothing in `src`, `tests` or the notebooks imports it.
  - From `pyproject.toml`: `python-bitvavo-api`, `rich` and `babel`. Only `dashboard.py` imports the last two, and Typer still pulls in `rich` itself.
  - Entries in `constants.py` that only the dashboard used.
- **`CLAUDE.md` rewritten:** the overview, current state, commands and architecture describe the MCP server, the exchange layer, the candle cache and the PRD instead of the Part A/B split. The notes on the model pipeline stay, because epic B builds on it.
- **`README.md` rewritten:** the Part A/B framing and the planned execution state machine are replaced by the MCP architecture, setup (`uv sync`, `fartt download`, `.mcp.json`, permissions) and a link to the PRD. Research-source sections stay where they still apply: risk-control sources for epic D, model sources for epic B.
- **Kept:** the three historical specs in `docs/specs/` that link to the deleted Part A PRD. They're records of past work.

## Quality gate

There is no CI, so lefthook is the gate. It follows the setup in the operator's `self-supervised-models` project, which closes the gap that pre-commit leaves: pre-commit only sees staged files, so a commit that touches only `pyproject.toml`, `uv.lock` or `.mcp.json` runs no tests and no type check.

- **pre-commit (kept):** notebook output stripping, `ruff format` and `ruff check --fix` on staged files, `pyright`, and `pytest`. A pytest exit code of 5 (no tests collected) counts as a pass.
- **pre-push (new):** the full suite on the whole repository, whatever the staged files were: `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright` and `uv run pytest`. A single push can skip it with `git push --no-verify`, deliberately and never by default.
- This lands as the first task commit of the first story pull request (#54), before any code changes, and `CLAUDE.md` describes it.

## Testing

- **Fake exchange** (`tests/fakes.py`): implements `Exchange` over an in-memory list of candles, with switches for a gap in the history and an unreachable exchange. Used by the cache and server tests.
- **`CcxtExchange`:** a stub ccxt client and a fixed clock cover the closed-candle filter, sorting and the unknown-market error, with no network.
- **`CandleCache`:** resume, no duplicates, skipping gaps, append-only writes, `latest()`, and filename conversion from `BTC/EUR` to `BTC-EUR`.
- **Server:** the SDK's in-memory client against `build_server(...)` covers structured output, the stale-data warning, both error paths and `count` validation. The exact v2 client API is checked when the plan is written.
- **Live smoke test:** fetches real BTC/EUR candles through ccxt. It's marked `network` and deselected by default in the pytest config, so the default suite stays offline. Run it with `uv run pytest -m network`.
- **Manual acceptance (#52):** a new Claude Code session in the project shows `fartt` as connected in `/mcp`, and the agent returns the latest candles when asked.

## Delivery

One pull request per story. This design lands in the first one, and later stories link to it on `main`.

| Pull request | Story | Commits |
|---|---|---|
| #54 | #50 exchange layer | PRD → this design → plan → pre-push gate → task commits |
| next | #51 candle cache and `fartt download` | plan → task commits |
| next | #52 MCP server and `get_candles` | plan → task commits |
| next | #53 removal and documentation | plan → task commits |

Every task commit passes the pre-commit hooks, and every push passes the pre-push gate.

## Out of scope

- Balances, orders, API keys, risk config: epics D and F.
- Forecast tools: epic C.
- A push-based trigger through Claude Code channels: future consideration in the PRD.
