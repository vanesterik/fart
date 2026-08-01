# Reuse Feature Pipeline as Model Input — Design

**Date:** 2026-08-01
**Status:** Approved
**Source:** [GitHub issue #4](https://github.com/vanesterik/fart/issues/4), part of [Epic #1: N-BEATS Signal Model](https://github.com/vanesterik/fart/issues/1)
**Related PRD:** `docs/product/prd/part-a-signal-generation-refactor.md` (Story 1)

## Problem

`fart/model/train_model.py` and `fart/model/predict_model.py` are currently empty stub files left over from the refactor. Before any regression model (N-BEATS or transformer, per Part A) can be trained, the project needs a working path from cached candle data to a feature DataFrame the model can consume — without changing the existing, working `calculate_technical_indicators.py` pipeline.

This is deliberately the first, narrowest story under Epic #1: prove the feature pipeline can feed training, not build the model itself.

## Acceptance Criteria (from issue #4)

- Given the existing `calculate_technical_indicators.py` pipeline produces a Polars DataFrame
- And given no changes are made to that pipeline
- When `train_model.py` loads features for training
- Then it consumes the DataFrame directly without transformation errors

## Architecture

### `fart/utils.py` — shared filepath convention

Add:

```python
def get_candle_filepath(settings: Settings) -> Path
```

Extracts the `data_dir / f"{market}-{interval}.csv"` naming convention that currently lives only inside `Downloader._determine_filepath`. Both `Downloader` (writer) and the new training loader (reader) must agree on the exact same filename; centralizing it avoids the two drifting apart. `Downloader._determine_filepath` is updated to call this helper instead of duplicating the format string.

### `fart/model/train_model.py` — currently empty stub

Add two functions:

```python
def prepare_training_data(
    settings: Settings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
```

Orchestrates the load → feature → split pipeline (see Data Flow below). This is the function that satisfies the acceptance criteria directly.

```python
def train(settings: Settings) -> None
```

The CLI-facing entry point. Calls `prepare_training_data` and logs the resulting train/test shapes via loguru (matching the logging style already used in `Downloader._log_settings`). Intentionally does **not** fit a model — that's Epic #1 Story 2/3 (N-BEATS / transformer training), which will extend this function once the target architecture's input shape is known.

### `fart/cli.py` — new `fart train` command

Mirrors the existing `download` command's option pattern:

```python
@app.command()
def train(
    data_dir: Optional[str] = typer.Option(None, help="..."),
    market: Optional[str] = typer.Option(None, help="..."),
    interval: Optional[str] = typer.Option(None, help="..."),
) -> None:
```

Builds a `Settings` instance via `update_settings` (same as `download`) and calls `train_model.train(settings)`. No `BITVAVO_API_KEY`/`BITVAVO_API_SECRET` env vars are required — this command only reads the local CSV cache, it never talks to Bitvavo.

## Data Flow

```
fart train
  → get_candle_filepath(settings)               # fart/utils.py
  → pl.read_csv(filepath)                        # raises FileNotFoundError if missing
  → calculate_technical_indicators(df)            # UNMODIFIED existing pipeline
  → df.drop_nulls()                               # removes indicator warm-up rows
  → train_test_split(df)                          # UNMODIFIED existing, non-shuffled, target=Close
  → (X_train, X_test, y_train, y_test)
  → shapes logged via loguru
```

Indicator warm-up produces leading `null`/`NaN` rows (e.g. Bollinger Bands period=20, MACD slow=26 + signal=9). These are dropped via `drop_nulls()` rather than imputed, keeping only real, fully-computed indicator values in the training data.

## Error Handling

The only new failure mode introduced by this story: the CSV for the requested market/interval doesn't exist yet (no `fart download` has been run). This raises `FileNotFoundError` with a message pointing the user at `fart download`.

`calculate_technical_indicators` and `train_test_split` are reused exactly as they exist today — no new error paths are introduced there, and no changes are made to either file.

## Testing

- Unit test for `get_candle_filepath` — verifies the path convention against known `Settings` values.
- Unit test for `prepare_training_data` — against a small synthetic Polars DataFrame with enough rows to survive indicator warm-up, asserting: no nulls survive into the splits, split shapes are internally consistent, and the pipeline runs end-to-end without raising.
- Unit test for the `FileNotFoundError` path when the CSV is missing.
- A CLI smoke test for `fart train` is left optional/out of scope — not required by the acceptance criteria.

## Out of Scope

- Sequence/window shaping for N-BEATS (or transformer) input — the exact lookback window size and stride depend on the architecture chosen in Epic #1 Story 2/3, and are deferred until then.
- Dropping or selecting feature columns (e.g. `Timestamp`) from `X_train`/`X_test` — the pipeline is consumed as-is per the acceptance criteria ("no changes... consumes the DataFrame directly").
- Any actual model fitting — `train()` logs shapes only; fitting a real model is Epic #1 Story 2/3.
- Filling in `predict_model.py` — that's Epic #1 Story 5.
- Fixing the pre-existing broken tests/imports noted in `CLAUDE.md` (`tests/model/test_train_test_split.py` imports `fart.common.constants`, which doesn't exist in the current layout) — unrelated to this story, not touched here.

## Open Questions

None — scope, data source, null handling, and CLI wiring were all confirmed during design.
