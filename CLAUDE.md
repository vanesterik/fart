# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FART (Financial Analysis & Real-time Trading) is a cryptocurrency trading platform that trains ML models on historical candle data from the Bitvavo exchange, generates buy/sell signals, and (eventually) executes trades in real time. It's a solo research project, not production infra — expect research code (notebooks) and production code (`src/fart`) to be at different levels of polish, and expect parts of `src/fart` to be genuinely incomplete rather than just "not yet read."

The project follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) layout: `notebooks/` for exploration, `references/` for supporting material, `src/fart` for library code, `tests/` for unit tests.

## Current state of the codebase (important)

The repo is mid-refactor. Recent commits (`BREAKING CHANGES`, `refactor: apply overhaul`) reorganized modules, and not everything downstream was updated to match:

- `src/fart/core/broker.py`, `src/fart/model/predict_model.py`, and `src/fart/model/train_model.py` are empty stub files.
- `src/fart/visualization/main.py` and `src/fart/visualization/candlestick_chart.py`'s surrounding imports reference modules that no longer exist in this layout (`fart.common`, `fart.utils.get_last_modified_data_file` as a submodule, `fart.visualization.plot_candlestick_chart`, `fart.common.feature_names`). The current, real layout is a flat `fart/utils.py` and `fart/constants.py`.
- `tests/utils/test_converters.py` imports `fart.utils.converters` and `tests/features/test_trade_strategy.py` imports `fart.common.constants` — neither exists in the current source tree. These tests do not currently pass/collect; don't assume the test suite is green before touching related code.
- `src/fart/core/dashboard.py` imports `rich` and `babel`, and `src/fart/visualization/candlestick_chart.py` imports `mplfinance` and `talib` — none of these are declared in `pyproject.toml` dependencies.

When working in these areas, check whether you're extending the intended new structure or patching dead code left over from the refactor — don't assume an import error means you made a mistake versus inheriting one.

`docs/product/` contains Problem Framing Canvas docs for two planned initiatives that explain *why* things are shaped the way they are:
- **Part A** — refactor the signal-generation model from classification (six classic ML classifiers predicting up/down/hold) to regression, using sequence-aware architectures (N-BEATS, time-series transformers), to capture trade magnitude vs. cost instead of a bare direction.
- **Part B** — build the trade execution system as a risk-management problem first (position sizing, stop-loss, kill switch, failure recovery) and a connector-building problem second, deliberately sequenced after Part A's signals are trustworthy.

## Documentation conventions

Never reference "superpowers" in code, file paths, or directory structure — e.g. no `docs/superpowers/...`. Specs and design docs live under `docs/specs/`.

## Commands

Dependency management is via `uv` (see `uv.lock`). The `Makefile` targets (`format`, `lint`, `requirements`, `test`, `run`, `data`, `visual`) wrap the `uv run ...` commands below.

```bash
uv sync                          # install dependencies
uv run fart download             # download candle data (see arguments below)
uv run pytest                    # run test suite
uv run pytest tests/features/test_trade_strategy.py::test_single_trade  # single test
uv run pytest -m "not slow"      # skip slow tests
uv run ruff format .             # format
uv run ruff check . --fix        # lint
uv run pyright                   # type check (strict mode, src/ only — tests/ excluded)
```

`fart download` takes three positional arguments, in order — `data_dir` (default `assets`), `market` (e.g. `BTC-EUR`, default `BTC-EUR`), `interval` (e.g. `1d`, `1h`, `30m` — see `Interval` enum in `settings.py`, default `1d`) — e.g. `uv run fart download assets BTC-EUR 1h`. Requires `BITVAVO_API_KEY` / `BITVAVO_API_SECRET` env vars (loaded via `.env` through `python-dotenv`); downloads are cached as CSV per market/interval under `data_dir` and resume from the last cached candle rather than re-fetching from scratch.

Pre-commit hooks are managed by `lefthook` (`.lefthook.yml`): notebooks get their outputs stripped, Python files get `ruff format` + `ruff check --fix`, then `pyright`, on every commit.

## User stories / issue tracking

User stories for this project are tracked as GitHub issues on the project board at https://github.com/users/vanesterik/projects/3. When asked to write a user story (or turn a piece of work into one), also create it as a GitHub issue in this repo and add it to that project board — don't just leave it as a local markdown file. Use the `gh` CLI, e.g.:

```bash
gh issue create --repo vanesterik/fart --title "..." --body "..."
gh project item-add 3 --owner vanesterik --url <issue-url>
```

## Architecture

- **`fart/settings.py`** — the `Settings` pydantic model (api credentials, data dir, market, interval) is the shared config object threaded through the CLI and downloader. `fart/utils.py:update_settings` merges CLI-supplied overrides onto a base `Settings` instance.
- **`fart/cli.py`** — Typer entrypoint (`fart` console script). Loguru is configured here (stderr + rotating `logs/cli.log`); commands build a `Settings` and hand it to a worker class (currently `Downloader`).
- **`fart/downloader.py`** — talks to Bitvavo's REST API (`python_bitvavo_api`) to backfill/append OHLCV candle data to a per-market/interval CSV cache, batching requests within Bitvavo's max-1440-candle-per-request limit and picking up from the last cached candle's timestamp on each run.
- **`fart/core/exchange.py`** — a typed wrapper around the untyped `python_bitvavo_api.Bitvavo` client (REST + websocket), exposing balance/price/trades/candles as pydantic models. `initiate()` opens a websocket subscription to candle updates and drives a callback; `wait_and_close()` blocks on Bitvavo's weight-based rate limiter before closing the socket.
- **`fart/core/dashboard.py`** — a `rich`-based live terminal dashboard (currency/balance/profit-loss panels) intended to render trading state during a live run; not currently wired into a working entrypoint (see "Current state" above).
- **`fart/features/`** — feature engineering over Polars DataFrames: `calculate_technical_indicators.py` computes Bollinger Bands / EMA / MACD / RSI via `talib`, configured through `technical_indicators_config.py`; `trade_strategy.py`'s `TradeStrategy` class backtests a buy/sell signal column against a DataFrame, tracking proceeds/trades/total return for strategy evaluation.
- **`fart/constants.py`** — centralized string labels (column names, UI labels) and a small color palette, shared across features, core, and visualization code to avoid stringly-typed column name drift between DataFrame producers and consumers.
- **`fart/model/`** — `train_test_split.py` does a non-shuffled (time-series-respecting) train/test split via sklearn; `train_model.py`/`predict_model.py` are currently empty stubs, to be filled per the Part A regression refactor.
- **`bruno/`** — a Bruno API collection for manually exercising the Bitvavo REST API (candles, markets) outside the app, useful when debugging what the Bitvavo client wrapper does under the hood.
