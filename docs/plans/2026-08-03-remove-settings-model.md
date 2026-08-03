# Remove Settings Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `Settings` pydantic model, its supporting `update_settings` helper, and the `Interval` enum, and have `Downloader` accept its configuration (`data_dir`, `market`, `interval`, `api_key`, `api_secret`) as plain constructor arguments instead of a `Settings` object — with `interval` as a plain `str` rather than an enum.

**Architecture:** `Downloader.__init__` currently takes a single `Settings` instance and reads `self._settings.<field>` throughout. It will instead take five explicit parameters and store them as individual `self._<field>` attributes. `fart/cli.py`'s `download` command currently builds a `Settings` via `update_settings` and passes it to `Downloader`; it will instead construct `Downloader` directly from its own CLI arguments (mirroring how the sibling `train` command already calls `train_model.train(data_dir=Path(data_dir), market=market, interval=interval)` with plain arguments — see `src/fart/cli.py:83`). `Settings` and `Interval` are both deleted from `fart/settings.py`; the unrelated `Candle` type alias defined in the same file is kept as-is since it is not part of the `Settings` model and is still consumed by `Downloader`. Everywhere `Downloader` currently type-hints or defaults against `Interval` (the constructor and the two timestamp-math helpers `_calculate_timestamp_list`/`_calculate_timestamp`) switches to plain `str`; behavior is unchanged since `Interval` was a `(str, Enum)` and every use of it (`.endswith(...)`, slicing, `.value`) already worked identically on a plain string. `update_settings` is deleted from `fart/utils.py`; `get_candle_filepath` and `get_last_modified_data_file` in that file are untouched.

**Tech Stack:** Python 3.11+, pydantic v2 (staying in the codebase for other models — only the `Settings` model itself goes), Typer CLI, `python_bitvavo_api`, pytest, `unittest.mock` (stdlib) for mocking the Bitvavo client in tests.

## Global Constraints

- Only remove `Settings`, `update_settings`, and `Interval`. Do not touch `Candle`, `get_candle_filepath`, or `get_last_modified_data_file` — they are unrelated to the `Settings` model and were not asked for.
- `Downloader`'s new constructor parameter order must be exactly: `data_dir, market, interval, api_key, api_secret` (per the request), with `interval: str` (not an enum).
- Match the existing sibling command style in `fart/cli.py`: the `download` command must build and pass raw arguments the same way the `train` command already does (`Path(data_dir)`, plain `market` and `interval` strings passed straight through — see rationale in Task 3).
- No new third-party dependencies. Use stdlib `unittest.mock` for the new downloader tests.
- Pre-commit (`lefthook`) runs `ruff format`, `ruff check --fix`, and `pyright` on every commit — code must pass all three before committing (pyright is strict mode, `src/` only, per `pyproject.toml`'s `[tool.pyright]`).
- Do not fix the pre-existing broken imports in `tests/utils/test_converters.py`, `tests/features/test_trade_strategy.py`, `src/fart/core/broker.py`, `src/fart/core/dashboard.py`, `src/fart/model/predict_model.py`, `src/fart/model/train_model.py` (train stub — not `prepare_training_data`), or `src/fart/visualization/*` — they are unrelated pre-existing breakage from an earlier refactor (documented in `CLAUDE.md`) and out of scope here.

---

## Current state (read this before starting)

`src/fart/settings.py` (32 lines):
```python
from enum import Enum
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel

Candle = Tuple[int, float, float, float, float, float]


class Interval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    ONE_WEEK = "1W"
    ONE_MONTH = "1M"


class Settings(BaseModel):
    api_key: str | None = None
    api_secret: str | None = None
    data_dir: Path = Path.home() / ".cache/fart"
    market: str = "BTC-EUR"
    interval: Interval = Interval.ONE_DAY
```

`src/fart/downloader.py` (169 lines) — full current content was read; the parts that change are `__init__`, `_validate_settings`, `_determine_filepath`, `_log_settings`, every `self._settings.<x>` reference inside `download()`, and the `interval: Interval` type hints/defaults on `_calculate_timestamp_list` and `_calculate_timestamp` (these two only change their type hint from `Interval` to `str` and their default from `Interval.ONE_DAY` to `"1d"` — their bodies already work unchanged on a plain string). `_convert_timestamp`, `_process_candles`, and the CSV helpers (`_load_cached_candle_data`, `_save_candle_data`, `_determine_start_timestamp`) reference neither `self._settings` nor `Interval` and are fully untouched.

`src/fart/cli.py`'s `download` command (`src/fart/cli.py:25-61`):
```python
@app.command()
def download(
    data_dir: Annotated[
        Optional[str],
        typer.Argument(
            help="Folder to save downloaded data (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        Optional[str],
        typer.Argument(
            help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC')."
        ),
    ] = "BTC-EUR",
    interval: Annotated[
        Optional[str],
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
) -> None:
    arguments = {
        "data_dir": data_dir,
        "market": market,
        "interval": interval,
    }
    settings = Settings(
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
    )
    settings = update_settings(
        settings=settings,
        arguments=arguments,
    )

    downloader = Downloader(settings)
    downloader.download()
```

`src/fart/utils.py` (46 lines) — `update_settings` (lines 7-20) is the only function that references `Settings`; `get_candle_filepath` and `get_last_modified_data_file` do not and stay.

No existing test currently imports `Settings`, `update_settings`, or constructs a `Downloader` — `grep -rln "update_settings" tests/` returns nothing, and there is no `tests/test_downloader.py` yet.

---

### Task 1: Give `Downloader` an explicit-arguments constructor

**Files:**
- Modify: `src/fart/downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Produces: `Downloader.__init__(self, data_dir: Path, market: str, interval: str, api_key: str | None, api_secret: str | None) -> None`, `Downloader.download(self) -> None` (signature unchanged), `Downloader._filepath: Path` (unchanged attribute name, still set in `__init__`).
- Consumes: only `Candle` from `fart.settings` after this task (unchanged import shape). `Settings` and `Interval` are no longer imported from there at all — both imports are removed in this task since neither is used in this file anymore, even though neither class/enum is deleted from `settings.py` until Task 2.

This task changes `Downloader`'s call signature and internals, including switching every `Interval`-typed parameter to plain `str`. `Settings` and `Interval` themselves are deleted from `fart/settings.py` in Task 2 — that's fine, because after this task `downloader.py` no longer imports or references either name at all, so the two tasks don't conflict regardless of order. Task 3 (updating `cli.py`) depends on this task being done first, since `cli.py` needs the new constructor shape to call it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_downloader.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from fart.downloader import Downloader


def _make_downloader(
    tmp_path: Path,
    mock_bitvavo: MagicMock,
    market: str = "BTC-EUR",
    known_markets: list[str] | None = None,
) -> Downloader:
    mock_bitvavo.return_value.markets.return_value = [
        {"market": m} for m in (known_markets if known_markets is not None else [market])
    ]
    return Downloader(
        data_dir=tmp_path,
        market=market,
        interval="1d",
        api_key="test-api-key",
        api_secret="test-api-secret",
    )


@patch("fart.downloader.Bitvavo")
def test_downloader_stores_configuration_and_computes_filepath(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    downloader = _make_downloader(tmp_path, mock_bitvavo, market="BTC-EUR")

    assert downloader._filepath == tmp_path / "BTC-EUR-1d.csv"
    assert tmp_path.exists()


@patch("fart.downloader.Bitvavo")
def test_downloader_passes_api_credentials_to_client(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    _make_downloader(tmp_path, mock_bitvavo)

    mock_bitvavo.assert_called_once_with(
        {"APIKEY": "test-api-key", "APISECRET": "test-api-secret"}
    )


@patch("fart.downloader.Bitvavo")
def test_downloader_unknown_market_raises(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="ETH-EUR"):
        _make_downloader(
            tmp_path, mock_bitvavo, market="ETH-EUR", known_markets=["BTC-EUR"]
        )


@patch("fart.downloader.Bitvavo")
def test_downloader_logs_configuration_without_leaking_secrets(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        _make_downloader(tmp_path, mock_bitvavo, market="BTC-EUR")
    finally:
        logger.remove(sink_id)

    logged = "\n".join(messages)
    assert "BTC-EUR" in logged
    assert "test-api-key" not in logged
    assert "test-api-secret" not in logged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_downloader.py -v`
Expected: FAIL — `TypeError: Downloader.__init__() got an unexpected keyword argument 'data_dir'` (current `__init__` only accepts `settings`).

- [ ] **Step 3: Rewrite `Downloader.__init__` and its helper methods**

In `src/fart/downloader.py`, replace the import line and the whole class header/`__init__`/`_validate_settings`/`_determine_filepath`/`_log_settings` block:

Replace:
```python
from fart.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.settings import Candle, Interval, Settings
from fart.utils import get_candle_filepath


class Downloader:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = Bitvavo(
            {
                "APIKEY": self._settings.api_key,
                "APISECRET": self._settings.api_secret,
            }
        )
        self._validate_settings()
        self._determine_filepath()
        self._log_settings()
```

With:
```python
from fart.constants import CLOSE, HIGH, LOW, OPEN, TIMESTAMP, VOLUME
from fart.settings import Candle
from fart.utils import get_candle_filepath


class Downloader:
    def __init__(
        self,
        data_dir: Path,
        market: str,
        interval: str,
        api_key: str | None,
        api_secret: str | None,
    ):
        self._data_dir = data_dir
        self._market = market
        self._interval = interval
        self._client = Bitvavo(
            {
                "APIKEY": api_key,
                "APISECRET": api_secret,
            }
        )
        self._validate_market()
        self._determine_filepath()
        self._log_configuration()
```

Replace:
```python
    def _validate_settings(self):
        market = self._settings.market
        markets = self._client.markets()  # type: ignore

        if not any(item["market"] == market for item in markets):
            raise ValueError(f"Market '{market}' not found in Bitvavo markets")

    def _determine_filepath(self):
        self._settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = get_candle_filepath(
            self._settings.data_dir,
            self._settings.market,
            self._settings.interval.value,
        )

    def _log_settings(self):
        settings_ = self._settings.model_dump()
        # Remove sensitive keys
        settings_.pop("api_key", None)
        settings_.pop("api_secret", None)
        # Add filepath for logging
        settings_["interval"] = self._settings.interval.value
        settings_["filepath"] = str(self._filepath)
        table = tabulate(settings_.items())
        logger.info(f"\n\nF.A.R.T. Downloader\n\n{table}\n")
```

With:
```python
    def _validate_market(self):
        markets = self._client.markets()  # type: ignore

        if not any(item["market"] == self._market for item in markets):
            raise ValueError(f"Market '{self._market}' not found in Bitvavo markets")

    def _determine_filepath(self):
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = get_candle_filepath(
            self._data_dir,
            self._market,
            self._interval,
        )

    def _log_configuration(self):
        configuration = {
            "data_dir": str(self._data_dir),
            "market": self._market,
            "interval": self._interval,
            "filepath": str(self._filepath),
        }
        table = tabulate(configuration.items())
        logger.info(f"\n\nF.A.R.T. Downloader\n\n{table}\n")
```

Then, inside `download()`, replace the three remaining `self._settings.<x>` reads:

Replace:
```python
        timestamp_list = self._calculate_timestamp_list(
            start_timestamp, interval=self._settings.interval
        )

        for start, end in tqdm(timestamp_list, desc="Downloading"):
            candles: List[Candle] = self._client.candles(  # type: ignore
                self._settings.market,
                self._settings.interval.value,
```

With:
```python
        timestamp_list = self._calculate_timestamp_list(
            start_timestamp, interval=self._interval
        )

        for start, end in tqdm(timestamp_list, desc="Downloading"):
            candles: List[Candle] = self._client.candles(  # type: ignore
                self._market,
                self._interval,
```

Everything below that (`start=`, `end=`, `_process_candles`, `_save_candle_data`) is unchanged — leave it exactly as-is.

Finally, drop the `Interval` type hint from the two timestamp-math helpers further down the file (their bodies are untouched — only the parameter's type annotation and default value change, since `Interval` was a `(str, Enum)` and these methods already only ever call plain-`str` operations on it):

Replace:
```python
    def _calculate_timestamp_list(
        self,
        start_timestamp: int,
        interval: Interval = Interval.ONE_DAY,
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> List[Tuple[int, int]]:
```

With:
```python
    def _calculate_timestamp_list(
        self,
        start_timestamp: int,
        interval: str = "1d",
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> List[Tuple[int, int]]:
```

Replace:
```python
    def _calculate_timestamp(
        self,
        timestamp: int,
        interval: Interval = Interval.ONE_DAY,
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> int:
```

With:
```python
    def _calculate_timestamp(
        self,
        timestamp: int,
        interval: str = "1d",
        epochs: int = 1440,  # Max limit per request set by Bitvavo
    ) -> int:
```

Neither method's body changes — `interval.endswith("m")`, `interval[:-1]`, etc. already operate identically on a plain `str`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_downloader.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Type-check and lint**

Run: `uv run pyright src/fart/downloader.py`
Expected: 0 errors.

Run: `uv run ruff check src/fart/downloader.py tests/test_downloader.py --fix && uv run ruff format src/fart/downloader.py tests/test_downloader.py`
Expected: clean, no remaining issues.

- [ ] **Step 6: Commit**

```bash
git add src/fart/downloader.py tests/test_downloader.py
git commit -m "refactor: make Downloader take explicit config args instead of Settings"
```

---

### Task 2: Delete `Settings`, `update_settings`, and `Interval`

**Files:**
- Modify: `src/fart/settings.py`
- Modify: `src/fart/utils.py`

**Interfaces:**
- Consumes: nothing new — this task only deletes code that Task 1 already stopped depending on (`downloader.py` no longer imports `Settings` or `Interval` after Task 1).
- Produces: `fart/settings.py` now only exports `Candle` (unchanged shape). `fart/utils.py` now only exports `get_candle_filepath` and `get_last_modified_data_file` (unchanged shapes/signatures).

This task has no behavior to test directly (pure deletion of otherwise-unused code) — verification is that the full suite still collects and passes, and pyright/ruff stay clean, which Step 3 covers. Do this task after Task 1: Task 1 removes `downloader.py`'s only remaining production dependency on `Settings` and `Interval`, so by the time this task runs, the only references left to delete are the definitions themselves and `cli.py`'s usage (handled in Task 3 — order between Task 2 and Task 3 doesn't matter functionally, but doing Task 2 first means Task 3 starts from a codebase where the old `Settings`-based CLI code is already a dangling reference, making the diff in Task 3 unambiguous).

- [ ] **Step 1: Remove the `Settings` class and `Interval` enum from `src/fart/settings.py`**

Replace the full file content:

```python
from typing import Tuple

Candle = Tuple[int, float, float, float, float, float]
```

(`enum.Enum`, `pathlib.Path`, and `pydantic.BaseModel` imports are all dropped along with `Settings` and `Interval` since nothing else in the file uses them — only `Candle` remains, and it only needs `Tuple`.)

- [ ] **Step 2: Remove `update_settings` from `src/fart/utils.py`**

Replace the full file content:

```python
from pathlib import Path


def get_candle_filepath(data_dir: Path, market: str, interval: str) -> Path:
    return data_dir / f"{market}-{interval}.csv"


def get_last_modified_data_file(data_dir: str) -> Path:
    """
    Get the last modified data file in the given directory.

    Parameters
    ----------
    - data_dir (Path): Path to the directory containing data files.

    Returns
    -------
    - Path: Path to the last modified data file.

    """

    # Get file list of csv data files in passed data directory
    file_list = list(Path(data_dir).glob("*.csv"))

    # Determine and return last modified data file of file list
    return max(file_list, key=lambda f: f.stat().st_mtime)
```

(The `Dict`/`Union` typing imports and the `Settings` import are dropped along with `update_settings`; `Path` stays since both remaining functions use it.)

- [ ] **Step 3: Verify nothing else references the deleted names**

Run: `grep -rn "Settings\|update_settings\|Interval" --include="*.py" src/ tests/`
Expected: no output (Task 3 hasn't touched `cli.py` yet, so if this still shows `src/fart/cli.py` matches, that's expected at this point in the plan and gets resolved in Task 3 — this check is here to confirm `settings.py` and `utils.py` themselves, and `downloader.py`, are clean).

Run: `uv run pytest tests/utils -v`
Expected: PASS (existing `test_get_candle_filepath.py` and `test_get_last_modified_data_file.py` are unaffected; `test_converters.py` was already broken before this plan per `CLAUDE.md` and is out of scope).

- [ ] **Step 4: Commit**

```bash
git add src/fart/settings.py src/fart/utils.py
git commit -m "refactor: remove Settings model, update_settings helper, and Interval enum"
```

---

### Task 3: Update `fart/cli.py`'s `download` command to build `Downloader` directly

**Files:**
- Modify: `src/fart/cli.py`

**Interfaces:**
- Consumes: `Downloader.__init__(self, data_dir: Path, market: str, interval: str, api_key: str | None, api_secret: str | None)` from Task 1.
- Produces: no new public interface — `download` remains a Typer command with the same CLI-visible argument names/help text/defaults.

The `train` command already shows the target pattern at `src/fart/cli.py:65-83`: plain `Annotated[str, typer.Argument(...)]` parameters, no `Optional`, converted straight into a `Path(data_dir)` call. Bring `download` in line with it instead of leaving it as the `Optional[str]` + dict-of-overrides style tied to `Settings`.

- [ ] **Step 1: Rewrite the `download` command and its imports**

Replace the import block:
```python
from fart.downloader import Downloader
from fart.model import train_model
from fart.settings import Settings
from fart.utils import update_settings
```

With:
```python
from fart.downloader import Downloader
from fart.model import train_model
```

Replace the `download` command body:
```python
@app.command()
def download(
    data_dir: Annotated[
        Optional[str],
        typer.Argument(
            help="Folder to save downloaded data (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        Optional[str],
        typer.Argument(
            help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC')."
        ),
    ] = "BTC-EUR",
    interval: Annotated[
        Optional[str],
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
) -> None:
    arguments = {
        "data_dir": data_dir,
        "market": market,
        "interval": interval,
    }
    settings = Settings(
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
    )
    settings = update_settings(
        settings=settings,
        arguments=arguments,
    )

    downloader = Downloader(settings)
    downloader.download()
```

With:
```python
@app.command()
def download(
    data_dir: Annotated[
        str,
        typer.Argument(
            help="Folder to save downloaded data (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        str,
        typer.Argument(
            help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC')."
        ),
    ] = "BTC-EUR",
    interval: Annotated[
        str,
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
) -> None:
    downloader = Downloader(
        data_dir=Path(data_dir),
        market=market,
        interval=interval,
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
    )
    downloader.download()
```

Note `Optional` becomes unused by this change — check whether `train`'s signature block (further down the file) still uses it before removing the import. It does not (`train` already uses plain `str`, not `Optional[str]`), so remove `Optional` from the `typing` import line at the top of the file:

Replace:
```python
from typing import Annotated, Optional
```

With:
```python
from typing import Annotated
```

- [ ] **Step 2: Type-check and lint**

Run: `uv run pyright src/fart/cli.py`
Expected: 0 errors.

Run: `uv run ruff check src/fart/cli.py --fix && uv run ruff format src/fart/cli.py`
Expected: clean, no remaining issues.

- [ ] **Step 3: Manually verify the CLI still works end-to-end**

Run: `uv run fart download --help`
Expected: help text lists `data_dir`, `market`, `interval` positional arguments with the same help strings as before (unchanged from the user's point of view).

Run: `uv run fart download /tmp/fart-plan-check BTC-EUR 1d`
Expected: either downloads/updates `/tmp/fart-plan-check/BTC-EUR-1d.csv` if `BITVAVO_API_KEY`/`BITVAVO_API_SECRET` are set in the environment/`.env`, or fails with an `Bitvavo` authentication/network error if they are not — either way, it must fail *after* constructing `Downloader` successfully (i.e. not with a `TypeError` about arguments or an `ImportError`/`AttributeError` about `Settings`). If it fails before that point, something in this task's rewrite is wrong — go back and check the constructor call matches Task 1's signature.

Clean up: `rm -rf /tmp/fart-plan-check`

- [ ] **Step 4: Commit**

```bash
git add src/fart/cli.py
git commit -m "refactor: build Downloader from CLI args directly, drop Settings usage"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass except the pre-existing, already-broken collection failures called out in `CLAUDE.md` (`tests/utils/test_converters.py`, `tests/features/test_trade_strategy.py`) — those are unrelated to this plan and must not be "fixed" as part of it. Every other test, including the new `tests/test_downloader.py`, must pass.

- [ ] **Step 2: Run pyright across the whole `src/` tree**

Run: `uv run pyright`
Expected: 0 errors (strict mode, `src/` only per `pyproject.toml`).

- [ ] **Step 3: Run ruff across the whole repo**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: clean, no remaining issues, no unexpected diffs outside the files this plan touched.

- [ ] **Step 4: Confirm `Settings`, `update_settings`, and `Interval` are fully gone**

Run: `grep -rn "Settings\|update_settings\|Interval" --include="*.py" src/ tests/`
Expected: no output.

---

## Self-review

**Spec coverage:**
- "Remove all implementations of the Settings model object defined in src/fart/settings.py" → Task 2, Step 1.
- "adjust src/fart/downloader.py to accept all arguments (data_dir, market, interval, api_key, api_secret) instead of a Settings object" → Task 1, exact parameter order matches.
- "Remove all Settings related functions and implementations" → Task 2 (`update_settings` in `utils.py`), Task 3 (`cli.py`'s `Settings(...)`/`update_settings(...)` call site), Task 4 Step 4 confirms via grep.
- "I notice you want to maintain the Interval class ... remove that as well and its corresponding implementations" → Task 1 switches `Downloader`'s constructor and its two timestamp-math helpers from `Interval` to plain `str`; Task 2, Step 1 deletes the `Interval` enum itself from `settings.py`; Task 3 drops the now-unneeded `Interval` import and `Interval(interval)` conversion from `cli.py`; Task 4 Step 4 confirms via grep.

**Placeholder scan:** every step has literal before/after code; no "similar to Task N", no TODOs.

**Type consistency:** `Downloader.__init__` parameter names/types (`data_dir: Path, market: str, interval: str, api_key: str | None, api_secret: str | None`) are identical across Task 1 (definition), Task 3 (call site in `cli.py`), and the test helper in Task 1's `tests/test_downloader.py`. `interval` flows as a plain string end-to-end — from the Typer CLI argument, through `cli.py`'s `Downloader(...)` call, into `Downloader`'s stored `self._interval`, and through to `_calculate_timestamp_list`/`_calculate_timestamp` — with no enum conversion anywhere.
