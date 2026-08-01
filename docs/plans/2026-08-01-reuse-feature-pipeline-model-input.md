# Reuse Feature Pipeline as Model Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fart/model/train_model.py` load the existing technical-indicator feature pipeline into a clean train/test split, wired up behind a new `fart train` CLI command, without modifying `calculate_technical_indicators.py` or `train_test_split.py`.

**Architecture:** A shared `get_candle_filepath` helper in `fart/utils.py` centralizes the CSV naming convention (used by both `Downloader` and the new loader). `fart/model/train_model.py` gains `prepare_training_data()` (load CSV → `calculate_technical_indicators` → drop warm-up nulls → `train_test_split`) and `train()` (CLI entry point that calls it and logs shapes). `fart/cli.py` gains a `train` command mirroring the existing `download` command's option pattern.

**Tech Stack:** Python 3.11+, Polars, pandas, scikit-learn (via existing `train_test_split.py`), TA-Lib (via existing `calculate_technical_indicators.py`), Typer, loguru, pytest, uv.

## Global Constraints

- Do not modify `src/fart/features/calculate_technical_indicators.py` or its computation logic in `src/fart/model/train_test_split.py` — reuse both unmodified (spec requirement).
- Never reference "superpowers" in code, file paths, or directory structure (CLAUDE.md).
- Indicator warm-up nulls are handled by dropping rows (`drop_nulls()`), not imputation (spec decision).
- `fart train` locates its input CSV the same way `fart download` writes it: `data_dir / f"{market}-{interval}.csv"`, derived from `Settings` (spec decision) — not `get_last_modified_data_file`.
- `fart train` requires no Bitvavo API credentials — it only reads the local CSV cache.
- New dependencies needed: `pandas`, `scikit-learn`, `ta-lib` are not yet declared in `pyproject.toml`. This plan declares them; it does **not** install the system TA-Lib C library — that's on the operator to install (e.g. `brew install ta-lib` on macOS) before `uv sync`/tests will succeed for Task 3 onward.
- `pyproject.toml`'s `[tool.pytest.ini_options] addopts` includes `--cov-fail-under=80` repo-wide. Scoped test runs in this plan (pointing at one test file) may still show a non-zero process exit due to that coverage gate even when every printed test result is `PASSED`. Treat the printed per-test PASSED/FAILED line as ground truth, not the process exit code.

---

## Task 1: Shared candle-file path helper

**Files:**
- Modify: `src/fart/utils.py`
- Test: `tests/utils/test_get_candle_filepath.py` (new)

**Interfaces:**
- Produces: `get_candle_filepath(settings: Settings) -> Path` — importable from `fart.utils`. Used by Task 2 (`Downloader`) and Task 3 (`prepare_training_data`).

- [ ] **Step 1: Write the failing test**

Create `tests/utils/test_get_candle_filepath.py`:

```python
from pathlib import Path

from fart.settings import Interval, Settings
from fart.utils import get_candle_filepath


def test_get_candle_filepath() -> None:
    settings = Settings(
        data_dir=Path("/tmp/fart-test-data"),
        market="BTC-EUR",
        interval=Interval.ONE_DAY,
    )

    filepath = get_candle_filepath(settings)

    assert filepath == Path("/tmp/fart-test-data/BTC-EUR-1d.csv")


def test_get_candle_filepath_different_market_and_interval() -> None:
    settings = Settings(
        data_dir=Path("/tmp/fart-test-data"),
        market="ETH-EUR",
        interval=Interval.ONE_HOUR,
    )

    filepath = get_candle_filepath(settings)

    assert filepath == Path("/tmp/fart-test-data/ETH-EUR-1h.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utils/test_get_candle_filepath.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_candle_filepath' from 'fart.utils'`

- [ ] **Step 3: Implement `get_candle_filepath`**

In `src/fart/utils.py`, insert after `update_settings` (after line 20, before the existing `get_last_modified_data_file` at line 23):

```python
def get_candle_filepath(settings: Settings) -> Path:
    market = settings.market
    interval = settings.interval.value
    return settings.data_dir / f"{market}-{interval}.csv"
```

No new imports needed — `Path` and `Settings` are already imported at the top of `src/fart/utils.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utils/test_get_candle_filepath.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fart/utils.py tests/utils/test_get_candle_filepath.py
git commit -m "feat: add get_candle_filepath helper for shared CSV path convention"
```

---

## Task 2: Point Downloader at the shared path helper

**Files:**
- Modify: `src/fart/downloader.py:11` (imports), `src/fart/downloader.py:55-60` (`_determine_filepath`)

**Interfaces:**
- Consumes: `get_candle_filepath(settings: Settings) -> Path` from Task 1.

No automated test is added for this task: `Downloader.__init__` constructs a real `python_bitvavo_api.Bitvavo` client and `_validate_settings()` calls it over the network, and there is currently no test coverage or mocking pattern for `Downloader` anywhere in the codebase to extend. Verification is via static checks plus a manual read-through, per the "no changes to filename convention" guarantee already covered by Task 1's test.

- [ ] **Step 1: Add the import**

In `src/fart/downloader.py`, change line 11:

```python
from fart.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.settings import Candle, Interval, Settings
```

to:

```python
from fart.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.settings import Candle, Interval, Settings
from fart.utils import get_candle_filepath
```

- [ ] **Step 2: Replace the inline path formula**

Replace `_determine_filepath` (lines 55-60):

```python
    def _determine_filepath(self):
        data_dir = self._settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        market = self._settings.market
        interval = self._settings.interval.value
        self._filepath = data_dir / f"{market}-{interval}.csv"
```

with:

```python
    def _determine_filepath(self):
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = get_candle_filepath(self._settings)
```

- [ ] **Step 3: Verify with static checks**

Run: `uv run ruff check src/fart/downloader.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 4: Commit**

```bash
git add src/fart/downloader.py
git commit -m "refactor: reuse get_candle_filepath in Downloader"
```

---

## Task 3: `prepare_training_data` — load, feature-engineer, split

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Modify: `src/fart/model/train_model.py` (currently empty)
- Test: `tests/model/test_train_model.py` (new)

**Interfaces:**
- Consumes: `get_candle_filepath(settings: Settings) -> Path` (Task 1); `calculate_technical_indicators(df: pl.DataFrame) -> pl.DataFrame` (existing, unmodified); `train_test_split(df: pl.DataFrame, target: str = CLOSE, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` (existing, unmodified).
- Produces: `prepare_training_data(settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` — importable from `fart.model.train_model`. Used by Task 4's `train()`.

- [ ] **Step 1: Declare the new dependencies**

Run:

```bash
uv add "pandas>=3.0.5" "scikit-learn>=1.9.0" "ta-lib>=0.7.1"
```

This updates `pyproject.toml` and `uv.lock` and installs into the environment. If it fails to build `ta-lib` because the system TA-Lib C library isn't installed, install it first (e.g. `brew install ta-lib` on macOS), then re-run the command above.

- [ ] **Step 2: Verify the new imports resolve**

Run: `uv run python -c "import talib, pandas, sklearn; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Write the failing tests**

Create `tests/model/test_train_model.py`:

```python
from pathlib import Path

import pytest

from fart.model.train_model import prepare_training_data
from fart.settings import Interval, Settings

CSV_HEADER = "Timestamp,Open,High,Low,Close,Volume\n"


def _write_candle_csv(path: Path, num_rows: int = 60) -> None:
    lines = [CSV_HEADER]
    base_price = 100.0
    for i in range(num_rows):
        timestamp = 1_600_000_000_000 + i * 86_400_000
        price = base_price + i
        lines.append(
            f"{timestamp},{price},{price + 1},{price - 1},{price},{1000 + i}\n"
        )
    path.write_text("".join(lines))


def test_prepare_training_data_returns_clean_split(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval=Interval.ONE_DAY,
    )
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv")

    X_train, X_test, y_train, y_test = prepare_training_data(settings)

    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
    assert not y_train.isnull().values.any()
    assert not y_test.isnull().values.any()

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] > 0
    assert X_train.shape[0] > X_test.shape[0]


def test_prepare_training_data_missing_csv_raises(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval=Interval.ONE_DAY,
    )

    with pytest.raises(FileNotFoundError):
        prepare_training_data(settings)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'prepare_training_data' from 'fart.model.train_model'`

- [ ] **Step 5: Implement `prepare_training_data`**

Write `src/fart/model/train_model.py`:

```python
from typing import Tuple

import pandas as pd
import polars as pl

from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.train_test_split import train_test_split
from fart.settings import Settings
from fart.utils import get_candle_filepath


def prepare_training_data(
    settings: Settings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    filepath = get_candle_filepath(settings)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)
    df = calculate_technical_indicators(df)
    df = df.drop_nulls()

    return train_test_split(df)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/fart/model/train_model.py tests/model/test_train_model.py
git commit -m "feat: load and split technical-indicator features for training"
```

---

## Task 4: `train()` — CLI-facing entry point with logging

**Files:**
- Modify: `src/fart/model/train_model.py`
- Test: `tests/model/test_train_model.py`

**Interfaces:**
- Consumes: `prepare_training_data(settings: Settings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` (Task 3).
- Produces: `train(settings: Settings) -> None` — importable from `fart.model.train_model`. Used by Task 5's `fart train` CLI command.

- [ ] **Step 1: Write the failing test**

In `tests/model/test_train_model.py`, change the import line:

```python
from fart.model.train_model import prepare_training_data
```

to:

```python
from loguru import logger

from fart.model.train_model import prepare_training_data, train
```

(`from loguru import logger` is a new top-level import; `train` is added to the existing `fart.model.train_model` import.)

Then append this test function to the end of the file:

```python
def test_train_logs_prepared_shapes(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval=Interval.ONE_DAY,
    )
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv")

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        train(settings)
    finally:
        logger.remove(sink_id)

    assert any("X_train" in message for message in messages)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model/test_train_model.py::test_train_logs_prepared_shapes -v`
Expected: FAIL with `ImportError: cannot import name 'train' from 'fart.model.train_model'`

- [ ] **Step 3: Implement `train`**

Replace the full contents of `src/fart/model/train_model.py` with:

```python
from loguru import logger
from typing import Tuple

import pandas as pd
import polars as pl

from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.train_test_split import train_test_split
from fart.settings import Settings
from fart.utils import get_candle_filepath


def prepare_training_data(
    settings: Settings,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    filepath = get_candle_filepath(settings)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)
    df = calculate_technical_indicators(df)
    df = df.drop_nulls()

    return train_test_split(df)


def train(settings: Settings) -> None:
    X_train, X_test, y_train, y_test = prepare_training_data(settings)
    logger.info(
        f"Prepared training data: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model/test_train_model.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/fart/model/train_model.py tests/model/test_train_model.py
git commit -m "feat: add train() entry point that logs prepared data shapes"
```

---

## Task 5: `fart train` CLI command

**Files:**
- Modify: `src/fart/cli.py`

**Interfaces:**
- Consumes: `train(settings: Settings) -> None` from `fart.model.train_model` (Task 4).

No automated CLI test is added — per the approved design spec this is optional/out of scope. Verification is a manual end-to-end run of the real command against a generated fixture CSV.

- [ ] **Step 1: Add the import**

In `src/fart/cli.py`, change line 9:

```python
from fart.downloader import Downloader
from fart.settings import Settings
from fart.utils import update_settings
```

to:

```python
from fart.downloader import Downloader
from fart.model import train_model
from fart.settings import Settings
from fart.utils import update_settings
```

- [ ] **Step 2: Add the `train` command**

Insert after the existing `download` command (after line 52's `downloader.download()`, before line 55's `if __name__ == "__main__":`):

```python
@app.command()
def train(
    data_dir: Optional[str] = typer.Option(
        None,
        help="Folder to load candle data from (defaults to system cache directory).",
    ),
    market: Optional[str] = typer.Option(
        None,
        help="Market to train on (e.g., 'BTC-EUR', 'BTC-USDC').",
    ),
    interval: Optional[str] = typer.Option(
        None,
        help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M').",
    ),
) -> None:
    arguments = {
        "data_dir": data_dir,
        "market": market,
        "interval": interval,
    }
    settings = update_settings(
        settings=Settings(),
        arguments=arguments,
    )

    train_model.train(settings)
```

- [ ] **Step 3: Manually verify end-to-end**

Generate a fixture CSV and run the command against it:

```bash
mkdir -p /tmp/fart-cli-smoke
python3 - <<'PYEOF'
from pathlib import Path

path = Path("/tmp/fart-cli-smoke/BTC-EUR-1d.csv")
lines = ["Timestamp,Open,High,Low,Close,Volume\n"]
base_price = 100.0
for i in range(60):
    timestamp = 1_600_000_000_000 + i * 86_400_000
    price = base_price + i
    lines.append(f"{timestamp},{price},{price + 1},{price - 1},{price},{1000 + i}\n")
path.write_text("".join(lines))
PYEOF

uv run fart train --data-dir /tmp/fart-cli-smoke --market BTC-EUR --interval 1d

rm -rf /tmp/fart-cli-smoke
```

Expected: the command exits without a traceback and stderr shows an INFO log line containing `Prepared training data: X_train=(...` with nonzero shapes.

- [ ] **Step 4: Static checks**

Run: `uv run ruff check src/fart/cli.py`
Expected: `All checks passed!`

Run: `uv run pyright`
Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 5: Commit**

```bash
git add src/fart/cli.py
git commit -m "feat: add fart train CLI command"
```

---

## Out of Scope (confirmed in spec, not covered by this plan)

- Sequence/window shaping for N-BEATS or transformer input (Epic #1 Story 2/3).
- Dropping or selecting feature columns (e.g. `Timestamp`) from `X_train`/`X_test`.
- Any actual model fitting — `train()` logs shapes only.
- Filling in `predict_model.py` (Epic #1 Story 5).
- Fixing pre-existing broken tests/imports noted in `CLAUDE.md` (e.g. `tests/model/test_train_test_split.py` importing `fart.common.constants`).
- Updating the stale `README.md`/`Makefile` `make train`/`make data` references — those describe a different (pre-CLI, script-invoked) mechanism that doesn't exist in the current layout and is unrelated to this story.
