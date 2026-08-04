# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FART (Financial Analysis & Real-time Trading) is a cryptocurrency trading platform that trains ML models on historical candle data from the Bitvavo exchange, generates buy/sell signals, and (eventually) executes trades in real time. It's a solo research project, not production infra — expect research code (notebooks) and production code (`src/fart`) to be at different levels of polish, and expect parts of `src/fart` to be genuinely incomplete rather than just "not yet read."

The project follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) layout: `notebooks/` for exploration, `references/` for supporting material, `src/fart` for library code, `tests/` for unit tests.

## Current state of the codebase (important)

The repo is mid-refactor. Recent commits reorganized modules and started the Part A regression rewrite, and not everything downstream was updated to match:

- `src/fart/model/train_model.py` implements `prepare_training_data` (loads cached candles, filters to a recent months slice, computes technical indicators, does a time-series train/test split) and `train` (fits `nbeats.py::NBeatsNet` on sliding return windows, returns per-candle magnitude/confidence). It's wired into the CLI as `fart train`. `src/fart/model/predict_model.py` and `src/fart/core/broker.py` are still empty stub files.
- `src/fart/core/exchange.py` (typed Bitvavo wrapper) and `src/fart/core/dashboard.py` (rich-based live dashboard) exist but aren't imported or wired into any entrypoint — `downloader.py` talks to the Bitvavo client directly instead of going through `Exchange`. They're most likely Part B trade-execution scaffolding waiting on `core/broker.py` to tie them together.
- `src/fart/visualization/main.py` and `src/fart/visualization/candlestick_chart.py`'s surrounding imports reference modules that no longer exist in this layout (`fart.common`, `fart.utils.get_last_modified_data_file` as a submodule, `fart.visualization.plot_candlestick_chart`, `fart.common.feature_names`). The current, real layout is a flat `fart/utils.py` and `fart/constants.py`. `main.py` also still uses `click`, inconsistent with the rest of the CLI's Typer pattern.
- `src/fart/core/dashboard.py` imports `rich` and `babel`, and `src/fart/visualization/candlestick_chart.py` imports `mplfinance` — none of these three are declared in `pyproject.toml` dependencies. (`ta-lib`/`talib` *is* declared and correctly used elsewhere, in `features/calculate_technical_indicators.py`.)
- The two test files that previously didn't collect (`tests/utils/test_converters.py`, `tests/features/test_trade_strategy.py`) have been deleted. `uv run pytest` currently collects and passes cleanly — don't assume the suite is still broken.

When working in these areas, check whether you're extending the intended new structure or patching dead code left over from the refactor — don't assume an import error means you made a mistake versus inheriting one.

`docs/product/` contains Problem Framing Canvas docs for two planned initiatives that explain *why* things are shaped the way they are:
- **Part A** — refactor the signal-generation model from classification (six classic ML classifiers predicting up/down/hold) to regression, using sequence-aware architectures (N-BEATS, time-series transformers), to capture trade magnitude vs. cost instead of a bare direction.
- **Part B** — build the trade execution system as a risk-management problem first (position sizing, stop-loss, kill switch, failure recovery) and a connector-building problem second, deliberately sequenced after Part A's signals are trustworthy.

## Documentation conventions

Never reference "superpowers" in code, file paths, or directory structure — e.g. no `docs/superpowers/...`. Specs and design docs live under `docs/specs/`.

## Library and framework syntax

Before writing or extending code against a library whose idiomatic patterns evolve over time (Typer, Pydantic, FastAPI, scikit-learn, etc.), use the `context7` MCP tool to check current syntax rather than relying on possibly-outdated recall. Also match the style of any existing sibling function/command in the same file (e.g. `fart/cli.py`'s Typer commands) rather than introducing a second, inconsistent pattern.

## Commands

Dependency management is via `uv` (see `uv.lock`). The `Makefile` targets (`format`, `lint`, `requirements`, `test`, `run`, `data`, `visual`) wrap the `uv run ...` commands below.

```bash
uv sync                          # install dependencies
uv run fart download             # download candle data (see arguments below)
uv run pytest                    # run test suite
uv run pytest tests/model/test_train_model.py::test_train_fits_nbeats_and_returns_magnitude_confidence  # single test
uv run pytest -m "not slow"      # skip slow tests
uv run ruff format .             # format
uv run ruff check . --fix        # lint
uv run pyright                   # type check (strict mode, src/ only — tests/ excluded)
```

`fart download` takes three positional arguments, in order — `data_dir` (default `assets`), `market` (e.g. `BTC-EUR`, default `BTC-EUR`), `interval` (e.g. `1d`, `1h`, `30m` — plain string, no local enum/validation, default `1d`) — e.g. `uv run fart download assets BTC-EUR 1h`. Requires `BITVAVO_API_KEY` / `BITVAVO_API_SECRET` env vars (loaded via `.env` through `python-dotenv`); downloads are cached as CSV per market/interval under `data_dir` and resume from the last cached candle rather than re-fetching from scratch.

`fart train` takes the same `data_dir`/`market`/`interval` positional arguments plus `--months` (most recent months of candle history to train on; trains on the complete cached history if not set), `--artifacts-dir` (default `artifacts`, where the trained model checkpoint is saved), and `--device` (`cpu`/`mps`, auto-detected if not set) — e.g. `uv run fart train assets BTC-EUR 1h --months 3`. Loads the cached CSV for that market/interval, computes technical indicators, time-series-splits it, then fits `NBeatsNet` (hyperparameters in `fart/model/nbeats_config.py::NBeatsConfig`) and logs per-candle magnitude/confidence — results are saved as a versioned model artifact (`fart/model/nbeats_persistence.py`) but not yet connected to a signal/predict path.

Pre-commit hooks are managed by `lefthook` (`.lefthook.yml`): notebooks get their outputs stripped, Python files get `ruff format` + `ruff check --fix`, then `pyright`, on every commit.

## Development workflow

For non-trivial work, ask before creating a feature branch, then work directly on it in this working directory (`git checkout -b <name>`) — do not use a git worktree or isolated agent workspace for this. Koen wants to review and test changes himself between steps in the same directory he's already using, which a separate worktree checkout makes inconvenient.

## User stories / issue tracking

User stories for this project are tracked as GitHub issues on the project board at https://github.com/users/vanesterik/projects/3. When asked to write a user story (or turn a piece of work into one), also create it as a GitHub issue in this repo and add it to that project board — don't just leave it as a local markdown file. Use the `gh` CLI, e.g.:

```bash
gh issue create --repo vanesterik/fart --title "..." --body "..."
gh project item-add 3 --owner vanesterik --url <issue-url>
```

## Architecture

- **`fart/cli.py`** — Typer entrypoint (`fart` console script). Loguru is configured here (stderr + rotating `logs/cli.log`). No shared config object — each command reads env vars / takes CLI args directly and calls its worker: `download` builds a `Downloader`, `train` calls `fart.model.train_model.train` directly.
- **`fart/downloader.py`** — talks to Bitvavo's REST API (`python_bitvavo_api`) directly via its own `Bitvavo` client (not through `core/exchange.py`) to backfill/append OHLCV candle data to a per-market/interval CSV cache, batching requests within Bitvavo's max-1440-candle-per-request limit and picking up from the last cached candle's timestamp on each run.
- **`fart/core/exchange.py`** — a typed wrapper around the untyped `python_bitvavo_api.Bitvavo` client (REST + websocket), exposing balance/price/trades/candles as pydantic models. `initiate()` opens a websocket subscription to candle updates and drives a callback; `wait_and_close()` blocks on Bitvavo's weight-based rate limiter before closing the socket. Not currently imported anywhere (see "Current state" above).
- **`fart/core/dashboard.py`** — a `rich`-based live terminal dashboard (currency/balance/profit-loss panels) intended to render trading state during a live run; not currently wired into a working entrypoint (see "Current state" above).
- **`fart/features/`** — feature engineering over Polars DataFrames: `calculate_technical_indicators.py` computes Bollinger Bands / EMA / MACD / RSI via `talib`, configured through `technical_indicators_config.py`; `trade_strategy.py`'s `TradeStrategy` class backtests a buy/sell signal column against a DataFrame, tracking proceeds/trades/total return for strategy evaluation.
- **`fart/constants.py`** — centralized string labels (column names, UI labels) and a small color palette, shared across features, core, and visualization code to avoid stringly-typed column name drift between DataFrame producers and consumers.
- **`fart/model/`** — Part A's regression pipeline. `train_test_split.py` does a non-shuffled (time-series-respecting) train/test split via sklearn. `train_model.py` (`prepare_training_data` + `train`) loads a market's cached candles, filters to a recent months slice, computes indicators, splits, builds sliding percent-return windows (`nbeats_dataset.py::build_return_windows`), and fits `nbeats.py::NBeatsNet` — a hand-rolled, doubly-residual, generic-basis N-BEATS net (stack of `NBeatsBlock`s, hyperparameters in `nbeats_config.py::NBeatsConfig`) trained with a Gaussian NLL loss to predict per-window mean/log-sigma, exposed as magnitude/confidence. Currently a quick prototype — results are logged, not persisted or connected to a predict path. `predict_model.py` is still an empty stub.