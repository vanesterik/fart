# Train N-BEATS on a Quick-Prototype Data Slice — Design

**Date:** 2026-08-03
**Status:** Approved
**Source:** [GitHub issue #5](https://github.com/vanesterik/fart/issues/5), part of [Epic #1: N-BEATS Signal Model](https://github.com/vanesterik/fart/issues/1)
**Related PRD:** `docs/product/part-a-signal-generation-refactor-prd.md` (Story 2, scoped down to the epic's "quick-prototype" tiny act of discovery)

## Problem

`fart/model/train_model.py` currently loads candle data, computes technical indicators, and applies the existing chronological train/test split — but `train()` only logs the resulting shapes; it doesn't fit a model. Before investing in full N-BEATS training/tuning (a later epic story), the project needs a fast sanity check: train N-BEATS on a small recent slice of BTC candles and confirm it produces a magnitude + confidence prediction per candle without errors.

This is deliberately narrow. It does not need to beat any baseline, run a real backtest, or produce a calibrated confidence value — those are separate, later stories (walk-forward backtest harness; select/wire up production model). It only needs to prove the architecture runs end-to-end on real data.

## Acceptance Criteria (from issue #5)

- Given a 6-month slice of historical BTC candles with features computed
- And given the existing time-series split is applied
- When the N-BEATS quick-prototype training script is run
- Then it outputs a magnitude + confidence prediction per candle without errors

## Architecture

### N-BEATS implementation: hand-rolled PyTorch, not a library

`torch` is already a bare dependency (`pyproject.toml`, CPU wheel index) but unused anywhere in `src/`. Rather than adopting a forecasting library (e.g. `neuralforecast`) with its own opinionated data/training API, N-BEATS is implemented directly as a small custom module, matching the codebase's existing pattern of small, focused modules under `fart/model/` and `fart/features/`.

### Univariate input: Close-price returns only

Classic N-BEATS is univariate — it forecasts a series from a lookback window of its own history. This prototype follows that directly: the model consumes a lookback window of Close-price percent returns, not the indicator columns (Bollinger, EMA, MACD, RSI) that Story 1 wired into `X_train`/`X_test`. `X_train`/`X_test` continue to be computed by `prepare_training_data()` (for API consistency and to leave the indicators available for a future multivariate story) but are not consumed by this model. Feeding indicators into N-BEATS as exogenous input is deliberately deferred — it's a genuinely separate design problem (feature scaling across very different indicator ranges, adapting the architecture to multivariate input) that would add failure surface to what's meant to be a fast sanity check.

### Magnitude: percent return, not raw price

`magnitude` = `(next_close - current_close) / current_close`, not a raw predicted price. BTC's price in this dataset spans roughly €3.4k (2019) to €55k (2026) — a return target is scale-invariant across that range and directly matches the PRD's framing of "trade magnitude" as the reason for moving off classification. `train_test_split()` itself is unmodified and still returns raw Close values as `y`; the return transform happens after the split, in the windowing step.

### Confidence: second output head via heteroscedastic regression

N-BEATS's per-block forecast head normally produces a single value (width = forecast horizon). Here the forecast head's output width is 2 instead of 1: `mu` (the magnitude prediction) and `log_sigma` (predicted log standard deviation), sharing the same backcast/forecast block stack. `confidence = 1 / (1 + exp(log_sigma))`, bounded in `(0, 1)` and decreasing as predicted uncertainty grows.

Training uses `torch.nn.GaussianNLLLoss(mu, target, var=exp(log_sigma) ** 2)` — the standard technique for learned uncertainty. This gives `confidence` a real training signal rather than being an untrained/meaningless placeholder number, while still not requiring the calibration work the PRD explicitly defers ("How is confidence calibrated/validated, not just magnitude?" — open question, not this story's job).

### Forecast horizon: 1 (next candle only)

The acceptance criteria asks for "a magnitude + confidence prediction per candle" (singular, per-candle) — this is single-step-ahead forecasting, not a multi-step forecast window. `forecast_length` is fixed at 1 throughout; only `lookback` (backcast length) is configurable.

### Only the final forecast is supervised

Per the original N-BEATS paper, the per-block backcast is architectural (drives the doubly-residual signal flow between stacked blocks) and is not separately supervised against the raw input. Only the summed final forecast (`mu`, `log_sigma`) is used in the loss, against the true next-step return.

## New Files

### `src/fart/model/nbeats_config.py`

Pydantic config, mirroring the existing style of `fart/features/technical_indicators_config.py`:

```python
class NBeatsConfig(BaseModel):
    lookback: int = 30            # backcast length, in candles
    num_stacks: int = 2
    num_blocks_per_stack: int = 3
    hidden_width: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
```

`forecast_length` is not part of the config — it's fixed at 1 (see above) and referenced as a module-level constant in `nbeats.py` rather than a tunable, since nothing in this story varies it.

### `src/fart/model/nbeats.py`

The network architecture only — no windowing, no training loop.

```python
class NBeatsBlock(nn.Module):
    """One generic N-BEATS block: FC stack -> backcast + forecast(width=2)."""
    def __init__(self, lookback: int, hidden_width: int) -> None: ...
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (backcast, forecast) where forecast has shape (batch, 2)."""

class NBeatsNet(nn.Module):
    """Stack of stacks of NBeatsBlock, doubly-residual, generic (non-interpretable) basis."""
    def __init__(self, config: NBeatsConfig) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, lookback) of returns. Returns (batch, 2) = (mu, log_sigma)."""
```

Doubly-residual stacking: each block receives the previous block's backcast residual (`x - backcast`) as its own input, and the final forecast is the sum of all blocks' forecasts — standard N-BEATS wiring, applied with `forecast width = 2` throughout instead of 1.

### `src/fart/model/nbeats_dataset.py`

Turns a chronological Close-price series into model-ready tensors. Kept separate from `nbeats.py` so the windowing/returns logic is testable independent of any torch model internals.

```python
def build_return_windows(
    close_prices: pl.Series,
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a chronological Close-price series, computes percent returns,
    then builds sliding windows of length `lookback` as model input,
    paired with the next return as the training target.

    Returns
    -------
    - X: FloatTensor of shape (num_windows, lookback)
    - y: FloatTensor of shape (num_windows,) — the next-step return
         immediately following each window
    """
```

`num_windows = len(close_prices) - lookback - 1` (one return is "lost" converting `len(close_prices)` prices to `len(close_prices) - 1` returns, then one more window-worth is needed as history before the first predictable target).

## Modified Files

### `src/fart/model/train_model.py`

`prepare_training_data()` gains a `months` parameter:

```python
def prepare_training_data(
    data_dir: Path,
    market: str,
    interval: str,
    months: int | None = 6,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
```

Immediately after `pl.read_csv(filepath)` and before `calculate_technical_indicators(df)`, if `months` is not `None`:

```python
cutoff = df[TIMESTAMP].max() - months * 30 * 24 * 60 * 60 * 1000  # 30-day months, Timestamp is epoch ms
df = df.filter(pl.col(TIMESTAMP) >= cutoff)
```

("Month" is approximated as 30 days — adequate for a prototype slice, avoids adding a date-arithmetic dependency for something that only needs to be roughly 6 months.)

Filtering happens before indicator calculation (not after) so indicators are computed only over the requested slice; the usual warm-up nulls (up to ~35 rows, from MACD's slow=26 + signal=9) are dropped from *within* that slice, same as today's `drop_nulls()` behavior over the full dataset.

`train()` is extended to actually fit and evaluate N-BEATS:

```python
def train(
    data_dir: Path,
    market: str,
    interval: str,
    months: int | None = 6,
    config: NBeatsConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

Steps:
1. Call `prepare_training_data(data_dir, market, interval, months)` → `X_train, X_test, y_train, y_test` (unchanged from today; `X_train`/`X_test` unused by this model, see Architecture above).
2. Build return-windows separately for train and test via `build_return_windows`, but computed from the full chronological Close series (`pl.concat([y_train, y_test])`) so returns chain correctly across the train/test boundary — then split the resulting windows back into train/test groups by index, preserving the same train/test row proportions as `train_test_split()` produced.
3. Instantiate `NBeatsNet(config or NBeatsConfig())`, train for `config.epochs` using `Adam(lr=config.learning_rate)` and `GaussianNLLLoss`.
4. Run inference (`model.eval()`, `torch.no_grad()`) over the test windows. `mu, log_sigma = model(X_test_windows).unbind(-1)`; `magnitudes = mu.numpy()`; `confidences = (1 / (1 + log_sigma.exp())).numpy()`.
5. Log a summary via loguru (matching the existing `X_train=... ` shape-logging style): candle count, mean/std of `magnitudes`, mean of `confidences`.
6. Return `(magnitudes, confidences)`.

### `src/fart/cli.py`

`train` command gains a `--months` option, default `6`, following the existing `Annotated[..., typer.Option(...)]` pattern already used elsewhere in the file:

```python
months: Annotated[
    int,
    typer.Option(help="How many months of the most recent candle history to train on."),
] = 6,
```

Passed through to `train_model.train(...)`.

## Data Flow

```
fart train --months 6
  → prepare_training_data(data_dir, market, interval, months=6)
      → pl.read_csv(filepath)
      → filter to Timestamp >= max(Timestamp) - 6*30 days
      → calculate_technical_indicators(df)     # UNMODIFIED existing pipeline
      → df.fill_nan(None).drop_nulls()
      → train_test_split(df)                    # UNMODIFIED, reused for its chronological split contract
      → (X_train, X_test, y_train, y_test)
  → build_return_windows(concat(y_train, y_test), lookback=30)
      → split windows back into train/test groups
  → NBeatsNet(NBeatsConfig()) fit via GaussianNLLLoss for `epochs`
  → inference on test windows
      → magnitudes (mu), confidences (1/(1+sigma))
  → log summary via loguru
  → return (magnitudes, confidences)
```

## Error Handling

No new failure modes beyond what `prepare_training_data()` already raises (`FileNotFoundError` when the CSV is missing). If the requested `months` slice is too small to produce even one training window after indicator warm-up is dropped (`num_windows <= 0`), `build_return_windows` raises `ValueError` with a message naming the required minimum row count — this is a real, reachable failure mode given how small a 6-month/1d slice is (~180 rows before warm-up, ~146 after), and the acceptance criteria explicitly requires the script to run "without errors" on a correctly-sized slice, so undersized input must fail loudly rather than silently producing zero windows.

## Testing

- `tests/model/test_nbeats_dataset.py` — `build_return_windows` on a small synthetic Close-price series with known values: verifies returned window/target shapes, verifies the return math against hand-computed percent changes, verifies the `ValueError` path when `lookback` exceeds the available data.
- `tests/model/test_nbeats.py` — `NBeatsNet` forward pass on a random input batch: asserts output shape `(batch, 2)`, asserts a single training step (forward + `GaussianNLLLoss` + `backward()`) runs without error and produces finite gradients.
- `tests/model/test_train_model.py` (extended) — a `months` filtering test (synthetic CSV spanning more than 6 months of daily rows, asserting rows outside the window are excluded), and an end-to-end `train()` test on a small synthetic CSV with `NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1)` passed explicitly to keep the test fast, asserting: no exception, `magnitudes`/`confidences` shapes match the test-window count, and every value in `confidences` is in `(0, 1)`.

## Out of Scope

- Multivariate input (feeding the indicator columns into N-BEATS) — deferred; `X_train`/`X_test` remain unused by this model.
- Any real evaluation of prediction quality (MAE/RMSE, backtest, comparison to the classifier baseline or buy-and-hold) — that's the walk-forward backtest harness story.
- Confidence calibration — explicitly an open question in the PRD, not addressed here; `confidence` only needs to exist and be trainable.
- `predict_model.py` — stays an empty stub; a stable prediction interface is Story 5's job, designed once both N-BEATS and the transformer exist.
- Multi-step forecasting (`forecast_length > 1`).
- GPU/accelerator support — `torch` is pinned to the CPU wheel index project-wide; this story doesn't change that.
- Hyperparameter tuning of `NBeatsConfig`'s defaults — the chosen values (lookback=30, 2 stacks × 3 blocks, hidden_width=64, 50 epochs) are reasonable starting points for a sanity check on ~150 rows of data, not a tuned result.

## Open Questions

None — architecture, input representation, magnitude/confidence definitions, data slicing, and output surfacing were all confirmed during design.
