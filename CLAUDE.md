# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FART (Financial Analysis & Real-time Trading) is a cryptocurrency trading platform that trains ML models on historical candle data from the Bitvavo exchange, generates buy/sell signals, and (eventually) executes trades in real time. It's a solo research project, not production infra — expect research code (notebooks) and production code (`src/fart`) to be at different levels of polish, and expect parts of `src/fart` to be genuinely incomplete rather than just "not yet read."

The project follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) layout: `notebooks/` for exploration, `docs/` for product framing and design docs, `src/fart` for library code, `tests/` for unit tests. There is no `references/` folder — it held an unused old-notebook/Bruno-collection grab-bag and was removed.

## Current state of the codebase (important)

The repo is mid-refactor. Part A (signal generation) is the active work; Part B (trade execution) is still just scaffolding:

- `src/fart/model/prepare_datasets.py::prepare_datasets` loads a market's cached CSV, sorts/deduplicates it by timestamp, computes `Magnitude` (signed percent-change), and builds sliding lag windows via its own `train_test_split` helper (non-shuffled, time-series-respecting split, distinct name from — and not to be confused with — the removed sklearn-based module of the same name). `src/fart/model/train_model.py::build_model` constructs a small two-hidden-layer `nn.Sequential` MLP regressor over a `num_lags`-wide window; `train_model` fits it with `nn.MSELoss`/Adam over minibatches (`init_dataloader`/`init_optimizer`). `src/fart/model/evaluate_model.py::evaluate_model` runs the trained model against both splits and reports directional accuracy (`calculate_accuracy`) and RMSE. `src/fart/model/persist_model.py::save_model`/`load_model` checkpoint the whole `nn.Module` (architecture + weights, via `torch.save`/`torch.load` — not just a `state_dict`, since this path has no separate config object to reconstruct the architecture from); `load_model` returns the model directly (in eval mode), not a tuple. All four are wired into the CLI as `fart train`.
- **This MLP is the first of five candidate architectures being screened**, not a committed final choice — see `docs/product/part-a-signal-generation-refactor-prd.md` §5/§7 for the full plan. `notebooks/2.0-kve-data-analysis.ipynb` (which builds/trains/evaluates the MLP) is the template each of CNN, GRU, N-BEATS, and a time-series transformer gets adapted from, reusing `prepare_datasets.py`/`evaluate_model.py`/`persist_model.py` where the architecture's input/output shape allows. Screening uses the same RMSE/directional-accuracy evaluation already built for the MLP; only the 1–2 architectures that screen best go on to a full walk-forward backtest. N-BEATS specifically was tried once before (`NBeatsNet`, `nbeats_config.py`, beta-NLL loss, a confidence/uncertainty output) and that implementation was deleted from the tree — a 130-run reproducibility check found beta-NLL didn't reliably fix its confidence-calibration problem (predicted confidence stayed only weakly informative, r ≈ −0.09, under every configuration tested). It's back in scope now as one of five candidates, being rebuilt fresh via the notebook-adaptation workflow above rather than restored from the deleted module — references to `NBeatsNet`/`nbeats_config`/`beta_nll_loss`/`confidence_calibration.py` in git history predate that deletion and don't exist in the current tree.
- `src/fart/model/predict_model.py` and `src/fart/core/broker.py` are still empty stub files — nothing downstream of a trained checkpoint (predictions, order placement, the live dashboard) is wired up yet. There's also no device selection in the current training path (`fart/model/device.py::get_device()` exists but nothing imports it) — training always runs on CPU.
- `src/fart/core/exchange.py` (typed Bitvavo wrapper) and `src/fart/core/dashboard.py` (rich-based live dashboard) exist but still aren't imported or wired into any entrypoint — `downloader.py` talks to the Bitvavo client directly instead of going through `Exchange`. They're Part B trade-execution scaffolding waiting on `core/broker.py` to tie them together; see `docs/product/part-b-trade-execution-system-prd.md` for the risk-management-first design that's supposed to gate this (position sizing, stop-loss, kill switch, failure handling *before* the connector) — none of it is built yet. `dashboard.py`'s `rich`/`babel` imports are declared in `pyproject.toml` dependencies.
- `fart/features/trade_strategy.py`'s `TradeStrategy` class has been replaced by `calculate_magnitude.py` (signed percent-change per candle) + `calculate_trade_returns.py` (a functional long-only backtest over a magnitude series, with cost/slippage/threshold/max-holding-period/stop-loss support and an optional `predicted_magnitudes` argument for backtesting a signal against realized price). `fart/visualization/` grew matching helpers (`magnitude.py`, `trade_returns.py`, `diverging_bar_chart.py`, `plot_styles.py`, `evaluation_line_chart.py`) alongside the existing `candlestick_chart.py` and `missing_value_heatmap.py`.
- The README (top-level, not this file) documents the Part A/B split, the actual working CLI commands, and a project structure tree — check there before re-deriving "why is it shaped this way" from scratch.

`uv run pytest` currently collects and passes cleanly — don't assume the suite is broken. When working in `core/` or `visualization/`, check whether you're extending the intended new structure or patching dead code left over from an earlier refactor — don't assume an import error means you made a mistake versus inheriting one.

`docs/product/` contains Problem Framing Canvas docs for two planned initiatives that explain *why* things are shaped the way they are:
- **Part A** — refactor the signal-generation model from classification (six classic ML classifiers predicting up/down/hold) to regression, to capture trade magnitude vs. cost instead of a bare direction. Five sequence-aware architectures (MLP, CNN, GRU, N-BEATS, time-series transformer) are screened on the same metrics before the best performer(s) go through full backtest validation — see the PRD for the two-stage plan.
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
uv run fart train                # train a model (see arguments below)
uv run pytest                    # run test suite
uv run pytest tests/model/test_train_model.py::test_train_model_reduces_loss  # single test
uv run pytest -m "not slow"      # skip slow tests
uv run ruff format .             # format
uv run ruff check . --fix        # lint
uv run pyright                   # type check (strict mode, src/ only — tests/ excluded)
```

`fart download` takes all arguments as options (no positionals) — `--assets-dir` (default `assets`), `--market` (e.g. `BTC-EUR`, default `BTC-EUR`), `--interval` (e.g. `1d`, `1h`, `30m` — plain string, no local enum/validation, default `1d`) — e.g. `uv run fart download --assets-dir assets --market BTC-EUR --interval 1h`. Requires `BITVAVO_API_KEY` / `BITVAVO_API_SECRET` env vars (loaded via `.env` through `python-dotenv`); downloads are cached as CSV per market/interval under `assets_dir` and resume from the last cached candle rather than re-fetching from scratch.

`fart train` takes the same `--assets-dir`/`--market`/`--interval` options plus `--artifacts-dir` (default `artifacts`, where the trained model checkpoint is saved), `--num-lags` (window size in past candles, default `100`), `--hidden-width` (width of the model's hidden layers, default `20`), `--batch-size` (default `16`), `--learning-rate` (default `1e-3`), `--num-epochs` (default `500`), and `--train-size` (proportion of windows used for training, default `0.8`) — e.g. `uv run fart train --assets-dir assets --market BTC-EUR --interval 1h --num-lags 50`. There is no `--months` filter (`prepare_datasets` trains on the complete cached history) and no `--device` flag (training always runs on CPU — see "Current state" above). Loads the cached CSV for that market/interval, sorts/deduplicates it, computes `Magnitude`, builds sliding lag windows, fits a small `nn.Sequential` MLP, logs directional accuracy/RMSE for both splits, and saves a versioned checkpoint via `persist_model.py` — not yet connected to a signal/predict path (`predict_model.py` is still an empty stub).

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

- **`fart/cli.py`** — Typer entrypoint (`fart` console script). Loguru is configured here (stderr + rotating `logs/cli.log`). No shared config object — each command reads env vars / takes CLI args directly: `download` builds a `Downloader`; `train` orchestrates `prepare_datasets.prepare_datasets` → `train_model.build_model`/`train_model` → `evaluate_model.evaluate_model` → `persist_model.save_model` directly, with no shared pipeline object.
- **`fart/downloader.py`** — talks to Bitvavo's REST API (`python_bitvavo_api`) directly via its own `Bitvavo` client (not through `core/exchange.py`) to backfill/append OHLCV candle data to a per-market/interval CSV cache, batching requests within Bitvavo's max-1440-candle-per-request limit and picking up from the last cached candle's timestamp on each run.
- **`fart/core/exchange.py`** — a typed wrapper around the untyped `python_bitvavo_api.Bitvavo` client (REST + websocket), exposing balance/price/trades/candles as pydantic models. `initiate()` opens a websocket subscription to candle updates and drives a callback; `wait_and_close()` blocks on Bitvavo's weight-based rate limiter before closing the socket. Not currently imported anywhere (see "Current state" above).
- **`fart/core/dashboard.py`** — a `rich`-based live terminal dashboard (currency/balance/profit-loss panels) intended to render trading state during a live run; not currently wired into a working entrypoint (see "Current state" above).
- **`fart/features/`** — feature engineering over Polars DataFrames: `calculate_technical_indicators.py` computes Bollinger Bands / EMA / MACD / RSI via `talib`, configured through `technical_indicators_config.py`; `calculate_magnitude.py` computes signed percent-change per candle; `calculate_trade_returns.py` is a functional long-only backtest over a magnitude series (cost/slippage/threshold/max-holding-period/stop-loss, with an optional `predicted_magnitudes` arg to backtest a signal known one tick ahead against realized price) — this replaced the older `trade_strategy.py::TradeStrategy` class.
- **`fart/constants.py`** — centralized string labels (column names, UI labels) and a small color palette, shared across features, core, and visualization code to avoid stringly-typed column name drift between DataFrame producers and consumers.
- **`fart/model/`** — Part A's regression pipeline. `prepare_datasets.py::prepare_datasets` loads a market's cached CSV, sorts/deduplicates it, computes `Magnitude`, and builds sliding lag windows via its own `train_test_split` helper (non-shuffled, time-series-respecting). `train_model.py::build_model` constructs a small two-hidden-layer `nn.Sequential` regressor; `train_model` fits it with `nn.MSELoss`/Adam over minibatches (`init_dataloader`/`init_optimizer`). `evaluate_model.py::evaluate_model` reports directional accuracy and RMSE on both splits. `persist_model.py` saves/loads the whole trained module via `torch.save`/`torch.load` (`load_model` returns the model directly, in eval mode). This replaced an earlier hand-rolled N-BEATS prototype (see "Current state" above) — `predict_model.py` (not yet connected to a signal path) is still an empty stub.
- **`fart/visualization/`** — Matplotlib/seaborn plotting helpers, mostly for notebook use. `candlestick_chart.py` (Bollinger/EMA/MACD/RSI overlays via `mplfinance`) and `missing_value_heatmap.py` predate Part A's regression pivot; `magnitude.py`, `trade_returns.py`, `diverging_bar_chart.py` (a shared bar-color-by-sign helper the last two build on), and `evaluation_line_chart.py` (a train/test/predicted line chart for evaluating a trained model, with an optional `y_train`) support it. `plot_styles.py::apply_plot_styles` sets shared tradingview-style `rcParams`.