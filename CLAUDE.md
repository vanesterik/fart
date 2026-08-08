# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FART (Financial Analysis & Real-time Trading) is a cryptocurrency trading platform that trains ML models on historical candle data from the Bitvavo exchange, generates buy/sell signals, and (eventually) executes trades in real time. It's a solo research project, not production infra — expect research code (notebooks) and production code (`src/fart`) to be at different levels of polish, and expect parts of `src/fart` to be genuinely incomplete rather than just "not yet read."

The project follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) layout: `notebooks/` for exploration, `docs/` for product framing and design docs, `src/fart` for library code, `tests/` for unit tests. There is no `references/` folder — it held an unused old-notebook/Bruno-collection grab-bag and was removed.

## Current state of the codebase (important)

The repo is mid-refactor. Part A (signal generation) is the active work and has moved past its first prototype; Part B (trade execution) is still just scaffolding:

- `src/fart/model/train_model.py` implements `prepare_training_data` (loads cached candles, filters to a recent months slice, computes technical indicators, does a time-series train/test split) and `train` (fits `nbeats.py::NBeatsNet` on sliding return windows, returns per-candle magnitude/confidence). It's wired into the CLI as `fart train`. Training now uses `nbeats_loss.py::beta_nll_loss` (Seitzer et al. 2022) instead of plain `nn.GaussianNLLLoss` — the original Gaussian NLL let the model shrink predicted confidence to a near-constant value regardless of actual error (a variance-collapse failure diagnosed and re-verified in `notebooks/2.0-kve-nbeats-confidence-calibration.ipynb`); beta-NLL's weighting (`NBeatsConfig.beta_nll`, default `0.5`) measurably improved that but confidence still isn't fully calibrated. `src/fart/model/predict_model.py` and `src/fart/core/broker.py` are still empty stub files — nothing downstream of a trained checkpoint (predictions, order placement, the live dashboard) is wired up yet.
- `src/fart/model/nbeats_persistence.py::load_model` returns `tuple[NBeatsNet, NBeatsConfig]` (not just the model) — a notebook or script written against an older single-value signature will silently misbehave (unpacking a tuple into `model` and then calling `model.blocks` etc. fails), not error at import time.
- `src/fart/core/exchange.py` (typed Bitvavo wrapper) and `src/fart/core/dashboard.py` (rich-based live dashboard) exist but still aren't imported or wired into any entrypoint — `downloader.py` talks to the Bitvavo client directly instead of going through `Exchange`. They're Part B trade-execution scaffolding waiting on `core/broker.py` to tie them together; see `docs/product/part-b-trade-execution-system-prd.md` for the risk-management-first design that's supposed to gate this (position sizing, stop-loss, kill switch, failure handling *before* the connector) — none of it is built yet.
- `src/fart/core/dashboard.py` imports `rich` and `babel`, neither declared in `pyproject.toml` dependencies (still true). `mplfinance` (used by `visualization/candlestick_chart.py`) **is** now declared, and that module's imports are otherwise clean — `fart/visualization/main.py` and the old `fart.common`/`fart.utils` submodule references it used to have are gone entirely, not just broken.
- `fart/features/trade_strategy.py`'s `TradeStrategy` class has been replaced by `calculate_magnitude.py` (signed percent-change per candle) + `calculate_trade_returns.py` (a functional long-only backtest over a magnitude series, with cost/slippage/threshold/max-holding-period/stop-loss support and an optional `predicted_magnitudes` argument for backtesting a signal against realized price). `fart/visualization/` grew matching helpers (`magnitude.py`, `trade_returns.py`, `diverging_bar_chart.py`, `plot_styles.py`) alongside the existing `candlestick_chart.py`, `missing_value_heatmap.py`, and `confidence_calibration.py`.
- The README (top-level, not this file) was substantially rewritten to match current reality — it now documents the Part A/B split, the actual working CLI commands, an accurate project structure tree, and a References section (split Part A/Part B) citing the papers/regulation behind both the signal-generation model and the planned trade-execution risk controls. Check there before re-deriving "why is it shaped this way" from scratch.

`uv run pytest` currently collects and passes cleanly — don't assume the suite is broken. When working in `core/` or `visualization/`, check whether you're extending the intended new structure or patching dead code left over from an earlier refactor — don't assume an import error means you made a mistake versus inheriting one.

`docs/product/` contains Problem Framing Canvas docs for two planned initiatives that explain *why* things are shaped the way they are:
- **Part A** — refactor the signal-generation model from classification (six classic ML classifiers predicting up/down/hold) to regression, using sequence-aware architectures (N-BEATS, time-series transformers), to capture trade magnitude vs. cost instead of a bare direction.
- **Part B** — build the trade execution system as a risk-management problem first (position sizing, stop-loss, kill switch, failure recovery) and a connector-building problem second, deliberately sequenced after Part A's signals are trustworthy.

## Documentation conventions

Never reference "superpowers" in code, file paths, or directory structure — e.g. no `docs/superpowers/...`. Specs and design docs live under `docs/specs/`.

## Library and framework syntax

Before writing or extending code against a library whose idiomatic patterns evolve over time (Typer, Pydantic, FastAPI, scikit-learn, etc.), use the `context7` MCP tool to check current syntax rather than relying on possibly-outdated recall. Also match the style of any existing sibling function/command in the same file (e.g. `fart/cli.py`'s Typer commands) rather than introducing a second, inconsistent pattern.

## Commands

Dependency management is via `uv` (see `uv.lock`). There is no `Makefile` — it was removed as redundant with `uv run ...` (documented directly below) and the `lefthook` pre-commit automation. Run these directly.

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
- **`fart/features/`** — feature engineering over Polars DataFrames: `calculate_technical_indicators.py` computes Bollinger Bands / EMA / MACD / RSI via `talib`, configured through `technical_indicators_config.py`; `calculate_magnitude.py` computes signed percent-change per candle; `calculate_trade_returns.py` is a functional long-only backtest over a magnitude series (cost/slippage/threshold/max-holding-period/stop-loss, with an optional `predicted_magnitudes` arg to backtest a signal known one tick ahead against realized price) — this replaced the older `trade_strategy.py::TradeStrategy` class.
- **`fart/constants.py`** — centralized string labels (column names, UI labels) and a small color palette, shared across features, core, and visualization code to avoid stringly-typed column name drift between DataFrame producers and consumers.
- **`fart/model/`** — Part A's regression pipeline. `train_test_split.py` does a non-shuffled (time-series-respecting) train/test split via sklearn. `train_model.py` (`prepare_training_data` + `train`) loads a market's cached candles, filters to a recent months slice, computes indicators, splits, builds sliding percent-return windows (`nbeats_dataset.py::build_return_windows`), and fits `nbeats.py::NBeatsNet` — a hand-rolled, doubly-residual, generic-basis N-BEATS net (stack of `NBeatsBlock`s, hyperparameters in `nbeats_config.py::NBeatsConfig`) trained with `nbeats_loss.py::beta_nll_loss` to predict per-window mean/log-sigma, exposed as magnitude/confidence. `nbeats_persistence.py` saves/loads versioned checkpoints (`load_model` returns `(model, config)`). Still a prototype — results are logged and checkpointed, but `predict_model.py` (not yet connected to a signal path) is an empty stub.
- **`fart/visualization/`** — Matplotlib/seaborn plotting helpers, mostly for notebook use. `candlestick_chart.py` (Bollinger/EMA/MACD/RSI overlays via `mplfinance`) and `missing_value_heatmap.py` predate Part A's regression pivot; `confidence_calibration.py`, `magnitude.py`, `trade_returns.py`, and `diverging_bar_chart.py` (a shared bar-color-by-sign helper the last two build on) support it. `plot_styles.py::apply_plot_styles` sets shared tradingview-style `rcParams`.