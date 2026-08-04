# Full N-BEATS Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fart train` work end-to-end against the complete cached candle history (not just a small recent slice), on both CPU and Apple Silicon MPS, and save each trained model as a versioned, reloadable artifact.

**Architecture:** `train_model.py`'s training loop switches from full-batch to minibatch (`torch.utils.data.DataLoader`/`TensorDataset`), so it scales to arbitrarily large window counts, and runs on whichever device `fart/model/device.py::get_device()` auto-detects (MPS if available, else CPU). At the end of every run, the fitted model is saved via new `fart/model/nbeats_persistence.py::save_model()` to a datetime-versioned path from new `fart/utils.py::get_model_filepath()`, so repeated runs accumulate distinct artifacts instead of overwriting. `fart/cli.py`'s `train` command gains `--full` (train on complete history), `--artifacts-dir`, and `--device` options.

**Tech Stack:** Python 3.11+, PyTorch 2.9 (CPU/MPS), Polars, NumPy, Pydantic, Typer, loguru, pytest, uv.

## Global Constraints

- `device` is a parameter on `train()`, not a field on `NBeatsConfig` — it's a runtime placement concern, not a model hyperparameter, and keeping it separate means it's never accidentally persisted as part of a saved artifact's config (spec decision).
- `get_device()` checks `torch.backends.mps.is_available()` only, never `torch.cuda.is_available()` — `torch` is pinned to the CPU-only wheel index (`pyproject.toml`) on Linux/Windows, so CUDA can never be available in this project; a CUDA branch would be permanently-dead code (spec decision).
- Model artifact filenames use a **microsecond**-resolution UTC timestamp prefix: `timestamp.strftime("%Y%m%dT%H%M%S%fZ")`. Second resolution was the original spec's plan, but it was found to collide empirically while verifying this plan — two `train()` calls within the same second silently overwrote each other. Do not regress to second resolution.
- `get_latest_model_filepath()` picks the max by **filename**, not filesystem mtime (mtime can be altered by copies/checkouts; the embedded timestamp can't) — matches the existing sibling `get_last_modified_data_file()`'s established style of an unguarded `max()` that raises `ValueError` on an empty result, rather than introducing a different error-handling convention (spec decision).
- `load_model()` always loads with `map_location="cpu"` — a saved artifact must stay hardware-portable regardless of what device trained it; callers move it to a device themselves if needed (spec decision). It also passes `weights_only=False` to `torch.load()` explicitly — verified necessary in this plan's research: the checkpoint bundles a plain Python dict (`config.model_dump()`) alongside the tensor `state_dict`, and being explicit here is the safer, verified choice rather than relying on `torch.load`'s changing defaults across versions.
- Before saving, the model is moved back to CPU (`model.cpu()`) so the saved `state_dict` always contains CPU tensors, consistent with `load_model()`'s CPU-only loading (spec decision).
- `artifacts_dir` is a required parameter on `train()` (no default) — same convention as the existing `data_dir`/`market`/`interval` params; only the CLI supplies a default (`"artifacts"`).
- `pyproject.toml`'s `[tool.pyright]` runs in strict mode over `src/` only (`tests/` excluded) — every task that touches `src/` must pass `uv run pyright` with 0 errors before committing. Verified in this plan's research: with torch 2.9.1, `DataLoader`/`TensorDataset`, `torch.save`/`torch.load`, `model.to(device)`, and `tensor.cpu()` all type-check cleanly under this repo's strict config with **no new** `# pyright: ignore` needed — the only pre-existing ignore in `train_model.py` (on `optimizer.step()`) is untouched.
- Never reference "superpowers" in code, file paths, or directory structure (CLAUDE.md).
- `pyproject.toml`'s `[tool.pytest.ini_options] addopts` includes `--cov-fail-under=80` repo-wide. Scoped test runs pointing at one test file may still show a non-zero process exit due to that coverage gate even when every printed test result is `PASSED`. Treat the printed per-test PASSED/FAILED line as ground truth, not the process exit code.
- No new dependencies are needed (verified in this plan's research: `uv.lock` shows the resolved macOS `torch` wheel has no `+cpu` suffix, unlike Linux/Windows, and already bundles MPS support) — no `uv add` required.

---

## Task 1: `NBeatsConfig` gains `batch_size`

**Files:**
- Modify: `src/fart/model/nbeats_config.py`
- Test: `tests/model/test_nbeats_config.py`

**Interfaces:**
- Produces: `NBeatsConfig.batch_size: int = 128` — new field on the existing pydantic model. Used by Task 5 (`train()`'s `DataLoader`s).

- [ ] **Step 1: Write the failing tests**

Append to `tests/model/test_nbeats_config.py`:

```python
def test_nbeats_config_default_batch_size() -> None:
    config = NBeatsConfig()

    assert config.batch_size == 128


def test_nbeats_config_batch_size_override() -> None:
    config = NBeatsConfig(batch_size=4)

    assert config.batch_size == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_nbeats_config.py -v`
Expected: `test_nbeats_config_default_batch_size` FAILs with `AttributeError: 'NBeatsConfig' object has no attribute 'batch_size'`

- [ ] **Step 3: Add the `batch_size` field**

In `src/fart/model/nbeats_config.py`, add a docstring line and the field:

```python
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
    - batch_size (int): Minibatch size for training and inference.

    """

    lookback: int = 30
    num_stacks: int = 2
    num_blocks_per_stack: int = 3
    hidden_width: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 128
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_nbeats_config.py -v`
Expected: `4 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/nbeats_config.py && uv run ruff check src/fart/model/nbeats_config.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/nbeats_config.py tests/model/test_nbeats_config.py
git commit -m "feat: add batch_size to NBeatsConfig"
```

---

## Task 2: `get_device()` — MPS/CPU auto-detection

**Files:**
- Create: `src/fart/model/device.py`
- Test: `tests/model/test_device.py` (new)

**Interfaces:**
- Produces: `get_device() -> torch.device` — importable from `fart.model.device`. Returns `torch.device("mps")` if `torch.backends.mps.is_available()`, else `torch.device("cpu")`. Used by Task 5 (`train()`).

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_device.py`. Patch target is `torch.backends.mps.is_available` directly (not an imported alias) — `device.py` calls it as `torch.backends.mps.is_available()`, an attribute access off the `torch` module itself, verified to work with this exact patch target:

```python
from unittest.mock import patch

from fart.model.device import get_device


@patch("torch.backends.mps.is_available", return_value=True)
def test_get_device_returns_mps_when_available(_mock_mps_available: object) -> None:
    device = get_device()

    assert device.type == "mps"


@patch("torch.backends.mps.is_available", return_value=False)
def test_get_device_returns_cpu_when_mps_unavailable(_mock_mps_available: object) -> None:
    device = get_device()

    assert device.type == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fart.model.device'`

- [ ] **Step 3: Implement `get_device`**

Create `src/fart/model/device.py`:

```python
import torch


def get_device() -> torch.device:
    """
    Auto-detect the best available torch device for this machine.

    Checks Apple Silicon MPS only, not CUDA -- `torch` is pinned to the
    CPU-only wheel index (`pyproject.toml`) on Linux/Windows, so
    `torch.cuda.is_available()` can never be True in this project.

    Returns
    -------
    - torch.device: `mps` if available, otherwise `cpu`.

    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_device.py -v`
Expected: `2 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/device.py && uv run ruff check src/fart/model/device.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/device.py tests/model/test_device.py
git commit -m "feat: add get_device for MPS/CPU auto-detection"
```

---

## Task 3: `get_model_filepath` and `get_latest_model_filepath`

**Files:**
- Modify: `src/fart/utils.py`
- Test: `tests/utils/test_get_model_filepath.py` (new)
- Test: `tests/utils/test_get_latest_model_filepath.py` (new)

**Interfaces:**
- Produces: `get_model_filepath(artifacts_dir: Path, market: str, interval: str, timestamp: datetime) -> Path` — importable from `fart.utils`. Returns `artifacts_dir / f"{timestamp:%Y%m%dT%H%M%S%fZ}-{market}-{interval}-nbeats.pt"`.
- Produces: `get_latest_model_filepath(artifacts_dir: Path, market: str, interval: str) -> Path` — importable from `fart.utils`. Globs `artifacts_dir` for `*-{market}-{interval}-nbeats.pt` and returns the max by filename. Raises `ValueError` if none match.
- Both used by Task 5 (`train()`, for saving) and available for future consumers (the not-yet-built backtest harness, #9) to look up the latest artifact.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_get_model_filepath.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from fart.utils import get_model_filepath


def test_get_model_filepath() -> None:
    timestamp = datetime(2026, 8, 4, 14, 47, 49, 32031, tzinfo=timezone.utc)

    filepath = get_model_filepath(
        artifacts_dir=Path("/tmp/fart-test-artifacts"),
        market="BTC-EUR",
        interval="1d",
        timestamp=timestamp,
    )

    assert filepath == Path(
        "/tmp/fart-test-artifacts/20260804T144749032031Z-BTC-EUR-1d-nbeats.pt"
    )


def test_get_model_filepath_different_market_and_interval() -> None:
    timestamp = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

    filepath = get_model_filepath(
        artifacts_dir=Path("/tmp/fart-test-artifacts"),
        market="ETH-EUR",
        interval="1h",
        timestamp=timestamp,
    )

    assert filepath == Path(
        "/tmp/fart-test-artifacts/20260101T000000000000Z-ETH-EUR-1h-nbeats.pt"
    )
```

Create `tests/utils/test_get_latest_model_filepath.py`:

```python
import tempfile
from pathlib import Path

import pytest

from fart.utils import get_latest_model_filepath


def test_get_latest_model_filepath_picks_max_by_filename() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        older = Path(temp_dir) / "20260101T000000000000Z-BTC-EUR-1d-nbeats.pt"
        newer = Path(temp_dir) / "20260804T144749032031Z-BTC-EUR-1d-nbeats.pt"
        other_market = Path(temp_dir) / "20261231T235959999999Z-ETH-EUR-1d-nbeats.pt"

        older.touch()
        newer.touch()
        other_market.touch()

        assert get_latest_model_filepath(Path(temp_dir), "BTC-EUR", "1d") == newer


def test_get_latest_model_filepath_no_matches_raises() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            get_latest_model_filepath(Path(temp_dir), "BTC-EUR", "1d")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/utils/test_get_model_filepath.py tests/utils/test_get_latest_model_filepath.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_model_filepath' from 'fart.utils'`

- [ ] **Step 3: Implement both functions**

In `src/fart/utils.py`, add the `datetime` import at the top and both new functions between `get_candle_filepath` and `get_last_modified_data_file`:

```python
from datetime import datetime
from pathlib import Path
```

```python
def get_model_filepath(
    artifacts_dir: Path, market: str, interval: str, timestamp: datetime
) -> Path:
    """
    Get the file path for a versioned N-BEATS model artifact. The datetime
    prefix means artifacts sort chronologically under a plain directory
    listing, and multiple training runs for the same market/interval don't
    overwrite each other.

    Parameters
    ----------
    - artifacts_dir (Path): Path to the directory to save model artifacts in.
    - market (str): Market name (e.g., 'BTC-USD').
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').
    - timestamp (datetime): Timestamp to prefix the file name with.

    Returns
    -------
    - Path: Path to the model artifact file.

    """
    prefix = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return artifacts_dir / f"{prefix}-{market}-{interval}-nbeats.pt"


def get_latest_model_filepath(artifacts_dir: Path, market: str, interval: str) -> Path:
    """
    Get the most recently trained N-BEATS model artifact for a market and
    interval, determined by the artifact file name's datetime prefix (not
    filesystem modification time, which copies/checkouts can alter).

    Parameters
    ----------
    - artifacts_dir (Path): Path to the directory model artifacts are saved in.
    - market (str): Market name (e.g., 'BTC-USD').
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').

    Returns
    -------
    - Path: Path to the most recent model artifact file.

    """
    file_list = list(artifacts_dir.glob(f"*-{market}-{interval}-nbeats.pt"))

    return max(file_list, key=lambda f: f.name)
```

(Insert both functions after `get_candle_filepath` and before `get_last_modified_data_file` — file order doesn't affect behavior, but keeps the two model-artifact functions adjacent to each other and near the candle-file convention they parallel.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/utils/test_get_model_filepath.py tests/utils/test_get_latest_model_filepath.py -v`
Expected: `4 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/utils.py && uv run ruff check src/fart/utils.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/utils.py tests/utils/test_get_model_filepath.py tests/utils/test_get_latest_model_filepath.py
git commit -m "feat: add get_model_filepath and get_latest_model_filepath"
```

---

## Task 4: `nbeats_persistence.py` — save/load a checkpoint

**Files:**
- Create: `src/fart/model/nbeats_persistence.py`
- Test: `tests/model/test_nbeats_persistence.py` (new)

**Interfaces:**
- Consumes: `NBeatsNet(config: NBeatsConfig)` and `NBeatsConfig` (both pre-existing, from `fart.model.nbeats` / `fart.model.nbeats_config`).
- Produces: `save_model(model: NBeatsNet, config: NBeatsConfig, path: Path) -> None` and `load_model(path: Path) -> NBeatsNet` — importable from `fart.model.nbeats_persistence`. Used by Task 5 (`train()`).

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_nbeats_persistence.py`:

```python
from pathlib import Path

import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_persistence import load_model, save_model


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=1, hidden_width=8
    )
    model = NBeatsNet(config)
    model.eval()

    path = tmp_path / "checkpoint.pt"
    save_model(model, config, path)
    loaded = load_model(path)

    assert loaded.state_dict().keys() == model.state_dict().keys()

    x = torch.randn(3, config.lookback)
    with torch.no_grad():
        original_output = model(x)
        loaded_output = loaded(x)

    assert torch.allclose(original_output, loaded_output)


def test_save_model_creates_missing_parent_directory(tmp_path: Path) -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=1, hidden_width=8
    )
    model = NBeatsNet(config)
    path = tmp_path / "nested" / "checkpoint.pt"

    save_model(model, config, path)

    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_nbeats_persistence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fart.model.nbeats_persistence'`

- [ ] **Step 3: Implement `save_model` and `load_model`**

Create `src/fart/model/nbeats_persistence.py`:

```python
from pathlib import Path
from typing import Any, Dict, cast

import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig


def save_model(model: NBeatsNet, config: NBeatsConfig, path: Path) -> None:
    """
    Save a trained NBeatsNet's weights and config as a single,
    self-describing checkpoint file.

    Parameters
    ----------
    - model (NBeatsNet): Trained model to save.
    - config (NBeatsConfig): Config the model was built and trained with.
    - path (Path): Destination file path. Parent directories are created if
      they don't exist.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": config.model_dump(),
    }
    torch.save(checkpoint, path)


def load_model(path: Path) -> NBeatsNet:
    """
    Load a saved NBeatsNet checkpoint, reconstructing the architecture from
    its bundled config before loading weights into it. Always loads onto
    CPU -- callers move the model to a specific device themselves if
    needed.

    Parameters
    ----------
    - path (Path): Path to a checkpoint previously saved by `save_model`.

    Returns
    -------
    - NBeatsNet: The reconstructed model, weights loaded, in eval mode.

    """
    checkpoint = cast(
        Dict[str, Any], torch.load(path, map_location="cpu", weights_only=False)
    )
    config = NBeatsConfig(**checkpoint["config"])
    model = NBeatsNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_nbeats_persistence.py -v`
Expected: `2 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/nbeats_persistence.py && uv run ruff check src/fart/model/nbeats_persistence.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/nbeats_persistence.py tests/model/test_nbeats_persistence.py
git commit -m "feat: add save_model/load_model for NBeatsNet checkpoints"
```

---

## Task 5: `train()` — minibatch training, device placement, versioned save

**Files:**
- Modify: `src/fart/model/train_model.py`
- Test: `tests/model/test_train_model.py`

**Interfaces:**
- Consumes: `NBeatsConfig.batch_size` (Task 1); `get_device() -> torch.device` (Task 2); `get_model_filepath(artifacts_dir, market, interval, timestamp) -> Path` (Task 3); `save_model(model, config, path) -> None` (Task 4).
- Produces: `train(data_dir: Path, market: str, interval: str, artifacts_dir: Path, months: Optional[int] = 6, config: Optional[NBeatsConfig] = None, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]` — `artifacts_dir` is new and required (no default); `device` is new and optional (`None` auto-detects via `get_device()`). This is the function `fart/cli.py` calls (Task 6).

- [ ] **Step 1: Update existing tests and write the new failing tests**

In `tests/model/test_train_model.py`, add `import torch` alongside the existing imports (after `import pytest`, before `from loguru import logger`):

```python
import numpy as np
import pytest
import torch
from loguru import logger
```

Update both existing `train(...)` calls to pass the new required `artifacts_dir` and pin `device=torch.device("cpu")` — required because `artifacts_dir` has no default, and pinning `device` keeps these tests deterministic regardless of what machine runs them (this repo's own dev machine has MPS available, and letting `get_device()` silently pick MPS in a test would make behavior depend on hardware).

Change `test_train_logs_prepared_shapes`:

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
            artifacts_dir=tmp_path / "artifacts",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
            device=torch.device("cpu"),
        )
    finally:
        logger.remove(sink_id)

    assert any("X_train" in message for message in messages)
```

Change `test_train_fits_nbeats_and_returns_magnitude_confidence`:

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
            artifacts_dir=tmp_path / "artifacts",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
            device=torch.device("cpu"),
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

Then append two new tests to the end of the file:

```python
def test_train_saves_versioned_artifact_each_run(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)
    artifacts_dir = tmp_path / "artifacts"
    config = NBeatsConfig(epochs=1, num_stacks=1, num_blocks_per_stack=1, batch_size=8)

    train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=artifacts_dir,
        config=config,
        device=torch.device("cpu"),
    )
    train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=artifacts_dir,
        config=config,
        device=torch.device("cpu"),
    )

    saved = sorted(artifacts_dir.glob("*-BTC-EUR-1d-nbeats.pt"))
    assert len(saved) == 2
    assert saved[0] != saved[1]


def test_train_minibatches_with_small_batch_size(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    magnitudes, confidences = train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=tmp_path / "artifacts",
        config=NBeatsConfig(
            epochs=1, num_stacks=1, num_blocks_per_stack=1, batch_size=4
        ),
        device=torch.device("cpu"),
    )

    assert magnitudes.shape[0] > 0
    assert np.all(np.isfinite(magnitudes))
    assert np.all(confidences > 0) and np.all(confidences < 1)
```

(200 daily rows, filtered to the default `months=6` inside `train()`, comfortably clears the default `lookback=30` window requirement after indicator warm-up is dropped, same row-count reasoning as the pre-existing tests. `batch_size=8`/`4` are deliberately smaller than the resulting window count, exercising the `DataLoader` batching path rather than a single full batch.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: The four `train(...)`-calling tests FAIL with `TypeError: train() got an unexpected keyword argument 'artifacts_dir'` (verified exact message against the current `train()` signature)

- [ ] **Step 3: Implement minibatch training, device placement, and versioned save**

Replace the full contents of `src/fart/model/train_model.py` with:

```python
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fart.constants import TIMESTAMP
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.features.sort_and_deduplicate_candles import sort_and_deduplicate_candles
from fart.model.device import get_device
from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.nbeats_persistence import save_model
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath, get_model_filepath


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
    df = sort_and_deduplicate_candles(df)

    if months is not None:
        max_timestamp = df[TIMESTAMP].max()
        assert isinstance(max_timestamp, int)
        cutoff = max_timestamp - months * 30 * 24 * 60 * 60 * 1000
        df = df.filter(pl.col(TIMESTAMP) >= cutoff)

    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)


def train(
    data_dir: Path,
    market: str,
    interval: str,
    artifacts_dir: Path,
    months: Optional[int] = 6,
    config: Optional[NBeatsConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    config = config or NBeatsConfig()
    device = device or get_device()

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

    model = NBeatsNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.GaussianNLLLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_windows, y_train_windows),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            mu, log_sigma = output.unbind(-1)
            loss = loss_fn(mu, y_batch, log_sigma.exp() ** 2)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType] -- Adam.step is untyped upstream (torch/optim/adam.py)
            epoch_loss += loss.item() * X_batch.shape[0]

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"loss={epoch_loss / len(X_train_windows):.6f}"
        )

    test_loader = DataLoader(
        TensorDataset(X_test_windows), batch_size=config.batch_size, shuffle=False
    )

    model.eval()
    mu_batches: List[Tensor] = []
    log_sigma_batches: List[Tensor] = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            mu, log_sigma = output.unbind(-1)
            mu_batches.append(mu.cpu())
            log_sigma_batches.append(log_sigma.cpu())

    magnitudes = torch.cat(mu_batches).numpy()
    confidences = (1 / (1 + torch.cat(log_sigma_batches).exp())).numpy()

    logger.info(
        f"N-BEATS: {len(magnitudes)} test candles, device={device}, "
        f"magnitude mean={magnitudes.mean():.5f} std={magnitudes.std():.5f}, "
        f"confidence mean={confidences.mean():.5f}"
    )

    timestamp = datetime.now(timezone.utc)
    model_path = get_model_filepath(artifacts_dir, market, interval, timestamp)
    save_model(model.cpu(), config, model_path)
    logger.info(f"Saved model artifact to {model_path}")

    return magnitudes, confidences
```

(`prepare_training_data` is unchanged from its current implementation — only `train()` changes.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: `8 passed`

- [ ] **Step 5: Static checks**

Run: `uv run ruff format src/fart/model/train_model.py tests/model/test_train_model.py && uv run ruff check src/fart/model/train_model.py tests/model/test_train_model.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 6: Commit**

```bash
git add src/fart/model/train_model.py tests/model/test_train_model.py
git commit -m "feat: minibatch training, device placement, and versioned model saving in train()"
```

---

## Task 6: `fart train --full` / `--artifacts-dir` / `--device` CLI options

**Files:**
- Modify: `src/fart/cli.py`

**Interfaces:**
- Consumes: `train_model.train(data_dir: Path, market: str, interval: str, artifacts_dir: Path, months: Optional[int] = 6, config: Optional[NBeatsConfig] = None, device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]` (Task 5).

No automated CLI test is added — matching this file's existing convention (the `download` and existing `train` commands have no CLI-level test, only manual verification). Verification here is a manual end-to-end run against a generated fixture CSV, with `--full`, `--artifacts-dir`, and `--device cpu` all exercised together.

- [ ] **Step 1: Add the new options**

In `src/fart/cli.py`, add `Optional` to the `typing` import and `torch` as a new import:

```python
import sys
from os import getenv
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
```

Replace the `train` command with:

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
            help="How many months of the most recent candle history to train on. Ignored if --full is set."
        ),
    ] = 6,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Train on the complete cached history instead of the recent --months slice.",
        ),
    ] = False,
    artifacts_dir: Annotated[
        str,
        typer.Option(help="Folder to save the trained model artifact to."),
    ] = "artifacts",
    device: Annotated[
        Optional[str],
        typer.Option(
            help="Torch device to train on ('cpu', 'mps'). Auto-detected if not set."
        ),
    ] = None,
) -> None:
    train_model.train(
        data_dir=Path(data_dir),
        market=market,
        interval=interval,
        artifacts_dir=Path(artifacts_dir),
        months=None if full else months,
        device=torch.device(device) if device else None,
    )
```

- [ ] **Step 2: Manually verify end-to-end**

Generate a fixture CSV large enough to clear the default `lookback=30` window requirement (200+ daily rows, per Task 5's row-count reasoning), then run the command against it with `--full`, `--artifacts-dir`, and `--device cpu`:

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

uv run fart train /tmp/fart-cli-smoke BTC-EUR 1d --full --artifacts-dir /tmp/fart-cli-smoke/artifacts --device cpu

ls /tmp/fart-cli-smoke/artifacts

rm -rf /tmp/fart-cli-smoke
```

Expected: the command exits without a traceback; stderr shows 50 `Epoch N/50: loss=...` INFO lines, a `N-BEATS: ... test candles, device=cpu, magnitude mean=... confidence mean=...` line, and a `Saved model artifact to /tmp/fart-cli-smoke/artifacts/<timestamp>-BTC-EUR-1d-nbeats.pt` line; the `ls` shows exactly one `.pt` file matching that name (this exact command was run during this plan's research and verified to produce this output).

- [ ] **Step 3: Static checks**

Run: `uv run ruff format src/fart/cli.py && uv run ruff check src/fart/cli.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 4: Commit**

```bash
git add src/fart/cli.py
git commit -m "feat: add --full, --artifacts-dir, --device options to fart train CLI command"
```

---

## Out of Scope (confirmed in spec, not covered by this plan)

- Actual integration with the backtest harness (#9) — only the save/load + latest-lookup contract.
- CUDA support — project-wide `pytorch-cpu` wheel pin means CUDA is unavailable on Linux/Windows; MPS/CPU only.
- Artifact cleanup/retention policy — versioned artifacts accumulate indefinitely in `artifacts/`; no pruning.
- Hyperparameter tuning of `NBeatsConfig` defaults for full-scale data quality — `epochs=50`, `hidden_width=64`, etc. stay as-is; `batch_size=128` is a reasonable starting default, not a tuned result.
- Streaming/lazy windowing — `build_return_windows` still builds all windows eagerly.
