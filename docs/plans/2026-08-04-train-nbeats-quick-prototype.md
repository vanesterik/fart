# Train N-BEATS on a Quick-Prototype Data Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fart train` actually fit a hand-rolled N-BEATS model on a recent slice of BTC candles and produce a magnitude + confidence prediction per held-out candle, without errors.

**Architecture:** A univariate N-BEATS network (`fart/model/nbeats.py`, `fart/model/nbeats_config.py`, `fart/model/nbeats_dataset.py`) consumes sliding windows of Close-price percent returns and predicts `(mu, log_sigma)` per window via a doubly-residual stack of generic blocks, trained with `GaussianNLLLoss`. `fart/model/train_model.py`'s `prepare_training_data()` gains a `months` slicing parameter, and `train()` is extended to build return-windows from the existing chronological split, fit the model, run inference on the held-out test windows, and return `(magnitudes, confidences)`. `fart/cli.py`'s `train` command gains a `--months` option.

**Tech Stack:** Python 3.11+, PyTorch (CPU), Polars, NumPy, Pydantic, Typer, loguru, pytest, uv.

## Global Constraints

- Do not modify `src/fart/features/calculate_technical_indicators.py` or `src/fart/model/train_test_split.py` — reuse both unmodified (spec requirement).
- N-BEATS input is univariate: Close-price percent returns only. `X_train`/`X_test` (the indicator columns) continue to be produced by `prepare_training_data()` but are not consumed by the model in this story (spec decision — multivariate input is deferred).
- `magnitude` = `(next_close - current_close) / current_close` (percent return), never a raw price (spec decision).
- `confidence = 1 / (1 + exp(log_sigma))`, must land in `(0, 1)` (spec decision).
- Forecast horizon is fixed at 1 (single-step-ahead, next candle only) — not configurable in this story.
- The N-BEATS forecast head's output width is 2 (`mu`, `log_sigma`), referenced as the module-level constant `FORECAST_WIDTH = 2` in `nbeats.py` — do not confuse this with the forecast horizon (which is 1).
- Only the final summed forecast is supervised (via `GaussianNLLLoss`); per-block backcast is architectural only, never given its own loss term (spec decision, matches the original N-BEATS paper).
- "Month" is approximated as 30 days when slicing by `months` — no date-arithmetic dependency needed (spec decision).
- `months` filtering is relative to the data's own most recent `Timestamp` (`df[TIMESTAMP].max()`), not wall-clock time — this keeps behavior correct even if the local CSV cache lags behind real time.
- `torch`, `numpy`, `pydantic`, and `polars` are already declared dependencies (`pyproject.toml`) — no new dependencies are needed for this plan, no `uv add` required.
- `pyproject.toml`'s `[tool.pyright]` runs in strict mode over `src/` only (`tests/` excluded) — every task that touches `src/` must pass `uv run pyright` with 0 errors before committing.
- Torch's type stubs are only partially typed: under this repo's strict pyright config, `nn.Module.__init__` (via `super().__init__()` in any `nn.Module` subclass) and `Optimizer.step()` both raise `reportUnknownMemberType`. This was verified directly against this repo's pyright/torch versions. Both call sites need a `# pyright: ignore[reportUnknownMemberType]` comment — see Task 3 and Task 5 below for the exact lines. No other torch call in this plan (`nn.Linear`, `nn.Sequential`, `nn.ModuleList`, `nn.ReLU`, `torch.zeros`, `torch.tensor`, `GaussianNLLLoss`, `.backward()`, `.numpy()`) needs one.
- Never reference "superpowers" in code, file paths, or directory structure (CLAUDE.md).
- `pyproject.toml`'s `[tool.pytest.ini_options] addopts` includes `--cov-fail-under=80` repo-wide. Scoped test runs pointing at one test file may still show a non-zero process exit due to that coverage gate even when every printed test result is `PASSED`. Treat the printed per-test PASSED/FAILED line as ground truth, not the process exit code.

---

## Task 1: `NBeatsConfig` — hyperparameter config

**Files:**
- Create: `src/fart/model/nbeats_config.py`
- Test: `tests/model/test_nbeats_config.py` (new)

**Interfaces:**
- Produces: `NBeatsConfig` (pydantic `BaseModel`) — importable from `fart.model.nbeats_config`. Fields: `lookback: int = 30`, `num_stacks: int = 2`, `num_blocks_per_stack: int = 3`, `hidden_width: int = 64`, `epochs: int = 50`, `learning_rate: float = 1e-3`. Used by Task 3 (`NBeatsNet`) and Task 5 (`train()`).

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_nbeats_config.py`:

```python
from fart.model.nbeats_config import NBeatsConfig


def test_nbeats_config_defaults() -> None:
    config = NBeatsConfig()

    assert config.lookback == 30
    assert config.num_stacks == 2
    assert config.num_blocks_per_stack == 3
    assert config.hidden_width == 64
    assert config.epochs == 50
    assert config.learning_rate == 1e-3


def test_nbeats_config_overrides() -> None:
    config = NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1)

    assert config.epochs == 2
    assert config.num_stacks == 1
    assert config.num_blocks_per_stack == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model/test_nbeats_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fart.model.nbeats_config'`

- [ ] **Step 3: Implement `NBeatsConfig`**

Create `src/fart/model/nbeats_config.py`:

```python
from pydantic import BaseModel


class NBeatsConfig(BaseModel):
    """
    Configuration for the quick-prototype N-BEATS model.

    Attributes
    ----------
    - lookback (int): Backcast length, in candles.
    - num_stacks (int): Number of stacks of blocks.
    - num_blocks_per_stack (int): Number of blocks per stack.
    - hidden_width (int): Width of each block's hidden fully-connected layers.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Adam optimizer learning rate.

    """

    lookback: int = 30
    num_stacks: int = 2
    num_blocks_per_stack: int = 3
    hidden_width: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model/test_nbeats_config.py -v`
Expected: `2 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/nbeats_config.py && uv run ruff check src/fart/model/nbeats_config.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/nbeats_config.py tests/model/test_nbeats_config.py
git commit -m "feat: add NBeatsConfig for quick-prototype N-BEATS hyperparameters"
```

---

## Task 2: `build_return_windows` — return-window dataset construction

**Files:**
- Create: `src/fart/model/nbeats_dataset.py`
- Test: `tests/model/test_nbeats_dataset.py` (new)

**Interfaces:**
- Produces: `build_return_windows(close_prices: pl.Series, lookback: int) -> Tuple[torch.Tensor, torch.Tensor]` — importable from `fart.model.nbeats_dataset`. Returns `(X, y)` where `X` has shape `(num_windows, lookback)` and `y` has shape `(num_windows,)`, both `torch.float32`. Raises `ValueError` if `num_windows <= 0`. Used by Task 5 (`train()`).

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_nbeats_dataset.py`:

```python
import polars as pl
import pytest
import torch

from fart.model.nbeats_dataset import build_return_windows


def test_build_return_windows_shapes_and_values() -> None:
    close_prices = pl.Series([100.0, 101.0, 99.0, 102.0, 104.0, 103.0])

    X, y = build_return_windows(close_prices, lookback=2)

    assert X.shape == (3, 2)
    assert y.shape == (3,)

    expected_returns = torch.tensor(
        [
            0.01,
            -0.019801980198019802,
            0.030303030303030304,
            0.0196078431372549,
            -0.009615384615384616,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(X[0], expected_returns[0:2], atol=1e-6)
    assert torch.allclose(X[1], expected_returns[1:3], atol=1e-6)
    assert torch.allclose(X[2], expected_returns[2:4], atol=1e-6)
    assert torch.allclose(y, expected_returns[2:5], atol=1e-6)


def test_build_return_windows_raises_when_too_few_prices() -> None:
    close_prices = pl.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError):
        build_return_windows(close_prices, lookback=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_nbeats_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fart.model.nbeats_dataset'`

- [ ] **Step 3: Implement `build_return_windows`**

Create `src/fart/model/nbeats_dataset.py`:

```python
from typing import Tuple

import numpy as np
import polars as pl
import torch


def build_return_windows(
    close_prices: pl.Series,
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Turns a chronological Close-price series into sliding windows of
    percent returns, paired with the next return as the training target.

    Parameters
    ----------
    - close_prices (pl.Series): Chronological Close prices.
    - lookback (int): Number of past returns per input window.

    Returns
    -------
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - X (torch.Tensor): Shape (num_windows, lookback), float32.
        - y (torch.Tensor): Shape (num_windows,), float32 — the next-step
          return immediately following each window.

    """
    prices = close_prices.to_numpy()
    returns = (prices[1:] - prices[:-1]) / prices[:-1]

    num_windows = len(returns) - lookback
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data to build a single window: need at least "
            f"{lookback + 2} close prices for lookback={lookback}, got {len(prices)}."
        )

    X = np.stack([returns[i : i + lookback] for i in range(num_windows)])
    y = returns[lookback : lookback + num_windows]

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_nbeats_dataset.py -v`
Expected: `2 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/nbeats_dataset.py && uv run ruff check src/fart/model/nbeats_dataset.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/nbeats_dataset.py tests/model/test_nbeats_dataset.py
git commit -m "feat: add build_return_windows for N-BEATS return-window construction"
```

---

## Task 3: `NBeatsNet` — the N-BEATS network

**Files:**
- Create: `src/fart/model/nbeats.py`
- Test: `tests/model/test_nbeats.py` (new)

**Interfaces:**
- Consumes: `NBeatsConfig` (Task 1).
- Produces: `NBeatsNet(config: NBeatsConfig)` (`torch.nn.Module`) — importable from `fart.model.nbeats`. `forward(x: Tensor) -> Tensor` where `x` has shape `(batch, config.lookback)` and the return has shape `(batch, 2)` (`[:, 0]` = `mu`, `[:, 1]` = `log_sigma`). Also produces `FORECAST_WIDTH = 2` module-level constant. Used by Task 5 (`train()`).

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_nbeats.py`:

```python
import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig


def test_nbeats_net_forward_output_shape() -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=2, hidden_width=8
    )
    model = NBeatsNet(config)
    x = torch.randn(4, config.lookback)

    output = model(x)

    assert output.shape == (4, 2)


def test_nbeats_net_training_step_produces_finite_gradients() -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=2, hidden_width=8
    )
    model = NBeatsNet(config)
    x = torch.randn(4, config.lookback)
    target = torch.randn(4)

    output = model(x)
    mu, log_sigma = output.unbind(-1)
    loss_fn = torch.nn.GaussianNLLLoss()
    loss = loss_fn(mu, target, log_sigma.exp() ** 2)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_nbeats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fart.model.nbeats'`

- [ ] **Step 3: Implement `NBeatsBlock` and `NBeatsNet`**

Create `src/fart/model/nbeats.py`:

```python
from typing import Tuple

import torch
from torch import Tensor, nn

from fart.model.nbeats_config import NBeatsConfig

FORECAST_WIDTH = 2


class NBeatsBlock(nn.Module):
    """One generic N-BEATS block: FC stack -> backcast + forecast(width=2)."""

    def __init__(self, lookback: int, hidden_width: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.fc = nn.Sequential(
            nn.Linear(lookback, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.backcast_layer = nn.Linear(hidden_width, lookback)
        self.forecast_layer = nn.Linear(hidden_width, FORECAST_WIDTH)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.fc(x)
        backcast = self.backcast_layer(hidden)
        forecast = self.forecast_layer(hidden)
        return backcast, forecast


class NBeatsNet(nn.Module):
    """Stack of NBeatsBlocks, doubly-residual, generic (non-interpretable) basis."""

    def __init__(self, config: NBeatsConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        num_blocks = config.num_stacks * config.num_blocks_per_stack
        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(config.lookback, config.hidden_width)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        forecast = torch.zeros(
            x.shape[0], FORECAST_WIDTH, dtype=x.dtype, device=x.device
        )
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_nbeats.py -v`
Expected: `2 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/nbeats.py && uv run ruff check src/fart/model/nbeats.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/nbeats.py tests/model/test_nbeats.py
git commit -m "feat: add hand-rolled NBeatsNet with mu/log_sigma forecast head"
```

---

## Task 4: `months` slicing in `prepare_training_data`

**Files:**
- Modify: `src/fart/model/train_model.py`
- Test: `tests/model/test_train_model.py`

**Interfaces:**
- Produces: `prepare_training_data(data_dir: Path, market: str, interval: str, months: Optional[int] = 6) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]` — the `months` param is new; existing 3-positional-arg call sites keep working via the default. Used by Task 5 (`train()`).

- [ ] **Step 1: Write the failing test**

In `tests/model/test_train_model.py`, add this test function (no new imports needed — `_write_candle_csv` already accepts a configurable `num_rows`):

```python
def test_prepare_training_data_filters_to_recent_months(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=400)

    X_train, X_test, y_train, y_test = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=6
    )
    total_rows_filtered = X_train.shape[0] + X_test.shape[0]

    X_train_full, X_test_full, y_train_full, y_test_full = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=None
    )
    total_rows_full = X_train_full.shape[0] + X_test_full.shape[0]

    assert total_rows_filtered < total_rows_full
    assert 100 <= total_rows_filtered <= 181
    assert total_rows_full > 300
```

(400 daily rows span far more than 6 months; the 6-month filter keeps the most recent ~181 calendar days before indicator warm-up rows are dropped, while the unfiltered call keeps close to all 400 minus warm-up. The bounds are loose on purpose — they check the filter meaningfully restricts the slice without hardcoding the exact indicator warm-up row count.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model/test_train_model.py::test_prepare_training_data_filters_to_recent_months -v`
Expected: FAIL with `TypeError: prepare_training_data() got an unexpected keyword argument 'months'`

- [ ] **Step 3: Implement `months` filtering**

In `src/fart/model/train_model.py`, add the import and update `prepare_training_data`:

```python
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from loguru import logger

from fart.constants import TIMESTAMP
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath


def prepare_training_data(
    data_dir: Path,
    market: str,
    interval: str,
    months: Optional[int] = 6,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    filepath = get_candle_filepath(data_dir, market, interval)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)

    if months is not None:
        cutoff = df[TIMESTAMP].max() - months * 30 * 24 * 60 * 60 * 1000
        df = df.filter(pl.col(TIMESTAMP) >= cutoff)

    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)


def train(data_dir: Path, market: str, interval: str) -> None:
    X_train, X_test, y_train, y_test = prepare_training_data(data_dir, market, interval)
    logger.info(
        f"Prepared training data: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )
```

(`train()` itself is untouched in this task — Task 5 extends it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: All tests pass, including the new `test_prepare_training_data_filters_to_recent_months`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/train_model.py && uv run ruff check src/fart/model/train_model.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/train_model.py tests/model/test_train_model.py
git commit -m "feat: filter prepare_training_data to a recent months slice"
```

---

## Task 5: `train()` fits N-BEATS and returns per-candle magnitude + confidence

**Files:**
- Modify: `src/fart/model/train_model.py`
- Test: `tests/model/test_train_model.py`

**Interfaces:**
- Consumes: `prepare_training_data(...)` (Task 4); `NBeatsConfig` (Task 1); `build_return_windows(close_prices: pl.Series, lookback: int) -> Tuple[torch.Tensor, torch.Tensor]` (Task 2); `NBeatsNet(config: NBeatsConfig)` (Task 3).
- Produces: `train(data_dir: Path, market: str, interval: str, months: Optional[int] = 6, config: Optional[NBeatsConfig] = None) -> Tuple[np.ndarray, np.ndarray]` — returns `(magnitudes, confidences)` over the held-out test windows. This is the function `fart/cli.py` calls (Task 6).

- [ ] **Step 1: Write the failing test**

In `tests/model/test_train_model.py`, add two new imports above the existing `from fart.model.train_model import prepare_training_data, train` line (the existing `from loguru import logger` import stays as-is, don't duplicate it):

```python
import numpy as np

from fart.model.nbeats_config import NBeatsConfig
```

`train()` now actually fits a model, which needs enough real data to build at least one `lookback=30` window after indicator warm-up and the train/test split — the pre-existing `test_train_logs_prepared_shapes` test's 60-row default fixture is too small for that (it would hit the `ValueError` from `build_return_windows`). Update it to use a larger fixture and a fast config, same reasoning as the new test below:

```python
def test_train_logs_prepared_shapes(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        train(
            data_dir=tmp_path,
            market="BTC-EUR",
            interval="1d",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
        )
    finally:
        logger.remove(sink_id)

    assert any("X_train" in message for message in messages)
```

Then append this new test function to the end of the file:

```python
def test_train_fits_nbeats_and_returns_magnitude_confidence(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        magnitudes, confidences = train(
            data_dir=tmp_path,
            market="BTC-EUR",
            interval="1d",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
        )
    finally:
        logger.remove(sink_id)

    assert isinstance(magnitudes, np.ndarray)
    assert isinstance(confidences, np.ndarray)
    assert magnitudes.shape == confidences.shape
    assert magnitudes.shape[0] > 0
    assert np.all(confidences > 0) and np.all(confidences < 1)
    assert any("X_train" in message for message in messages)
```

(200 daily rows, filtered to the default `months=6` inside `train()`, comfortably clears the default `lookback=30` window requirement after indicator warm-up is dropped — see Task 4's row-count math. `epochs=2` and single stack/block keep the test fast.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model/test_train_model.py::test_train_fits_nbeats_and_returns_magnitude_confidence -v`
Expected: FAIL — `train()` currently returns `None` and takes no `config` argument (`TypeError: train() got an unexpected keyword argument 'config'`)

- [ ] **Step 3: Implement the N-BEATS fit + inference in `train()`**

Replace the full contents of `src/fart/model/train_model.py` with:

```python
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn

from fart.constants import TIMESTAMP
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath


def prepare_training_data(
    data_dir: Path,
    market: str,
    interval: str,
    months: Optional[int] = 6,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    filepath = get_candle_filepath(data_dir, market, interval)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)

    if months is not None:
        cutoff = df[TIMESTAMP].max() - months * 30 * 24 * 60 * 60 * 1000
        df = df.filter(pl.col(TIMESTAMP) >= cutoff)

    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)


def train(
    data_dir: Path,
    market: str,
    interval: str,
    months: Optional[int] = 6,
    config: Optional[NBeatsConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    config = config or NBeatsConfig()

    X_train, X_test, y_train, y_test = prepare_training_data(
        data_dir, market, interval, months
    )
    logger.info(
        f"Prepared training data: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )

    n_train = y_train.shape[0]
    close_prices = pl.concat([y_train, y_test])
    X_all, y_all = build_return_windows(close_prices, config.lookback)

    n_train_windows = max(0, n_train - config.lookback - 1)
    X_train_windows, y_train_windows = X_all[:n_train_windows], y_all[:n_train_windows]
    X_test_windows = X_all[n_train_windows:]

    model = NBeatsNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.GaussianNLLLoss()

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        output = model(X_train_windows)
        mu, log_sigma = output.unbind(-1)
        loss = loss_fn(mu, y_train_windows, log_sigma.exp() ** 2)
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    model.eval()
    with torch.no_grad():
        output = model(X_test_windows)
        mu, log_sigma = output.unbind(-1)
        magnitudes = mu.numpy()
        confidences = (1 / (1 + log_sigma.exp())).numpy()

    logger.info(
        f"N-BEATS quick prototype: {len(magnitudes)} test candles, "
        f"magnitude mean={magnitudes.mean():.5f} std={magnitudes.std():.5f}, "
        f"confidence mean={confidences.mean():.5f}"
    )

    return magnitudes, confidences
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: All tests pass, including `test_train_fits_nbeats_and_returns_magnitude_confidence` and the updated `test_train_logs_prepared_shapes`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/train_model.py && uv run ruff check src/fart/model/train_model.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/train_model.py tests/model/test_train_model.py
git commit -m "feat: fit NBeatsNet in train() and return per-candle magnitude/confidence"
```

---

## Task 6: `fart train --months` CLI option

**Files:**
- Modify: `src/fart/cli.py`

**Interfaces:**
- Consumes: `train_model.train(data_dir: Path, market: str, interval: str, months: Optional[int] = 6, config: Optional[NBeatsConfig] = None) -> Tuple[np.ndarray, np.ndarray]` (Task 5).

No automated CLI test is added — matching this file's existing convention (the `download` and prior `train` commands have no CLI-level test, only manual verification). Verification here is a manual end-to-end run against a generated fixture CSV.

- [ ] **Step 1: Add the `--months` option**

In `src/fart/cli.py`, replace the `train` command with:

```python
@app.command()
def train(
    data_dir: Annotated[
        str,
        typer.Argument(
            help="Folder to load candle data from (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        str,
        typer.Argument(help="Market to train on (e.g., 'BTC-EUR', 'BTC-USDC')."),
    ] = "BTC-EUR",
    interval: Annotated[
        str,
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
    months: Annotated[
        int,
        typer.Option(
            help="How many months of the most recent candle history to train on."
        ),
    ] = 6,
) -> None:
    train_model.train(
        data_dir=Path(data_dir), market=market, interval=interval, months=months
    )
```

- [ ] **Step 2: Manually verify end-to-end**

Generate a fixture CSV large enough to clear the default `lookback=30` window requirement after the default 6-month filter and indicator warm-up (per Task 4/5's row-count math, 200+ daily rows is comfortably enough), then run the command against it:

```bash
mkdir -p /tmp/fart-cli-smoke
python3 - <<'PYEOF'
from pathlib import Path

path = Path("/tmp/fart-cli-smoke/BTC-EUR-1d.csv")
lines = ["Timestamp,Open,High,Low,Close,Volume\n"]
base_price = 100.0
for i in range(200):
    timestamp = 1_600_000_000_000 + i * 86_400_000
    price = base_price + i
    lines.append(f"{timestamp},{price},{price + 1},{price - 1},{price},{1000 + i}\n")
path.write_text("".join(lines))
PYEOF

uv run fart train /tmp/fart-cli-smoke BTC-EUR 1d --months 6

rm -rf /tmp/fart-cli-smoke
```

Expected: the command exits without a traceback; stderr shows an INFO log line containing `Prepared training data: X_train=(...` and a second INFO line containing `N-BEATS quick prototype: ... test candles, magnitude mean=... confidence mean=...`.

- [ ] **Step 3: Static checks**

Run: `uv run ruff format src/fart/cli.py && uv run ruff check src/fart/cli.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 4: Commit**

```bash
git add src/fart/cli.py
git commit -m "feat: add --months option to fart train CLI command"
```

---

## Out of Scope (confirmed in spec, not covered by this plan)

- Multivariate input (feeding the indicator columns into N-BEATS) — `X_train`/`X_test` remain unused by this model.
- Any real evaluation of prediction quality (MAE/RMSE, backtest, comparison to the classifier baseline or buy-and-hold).
- Confidence calibration.
- Filling in `predict_model.py`.
- Multi-step forecasting (`forecast_length > 1`).
- GPU/accelerator support.
- Hyperparameter tuning of `NBeatsConfig`'s defaults.
