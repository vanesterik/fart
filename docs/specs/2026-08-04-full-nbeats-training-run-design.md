# Full N-BEATS Training Run — Design

**Date:** 2026-08-04
**Status:** Approved
**Source:** [GitHub issue #6](https://github.com/vanesterik/fart/issues/6), part of [Epic #1: N-BEATS Signal Model](https://github.com/vanesterik/fart/issues/1)
**Related PRD:** `docs/product/part-a-signal-generation-refactor-prd.md` (Story 2)
**Builds on:** `docs/specs/2026-08-03-train-nbeats-quick-prototype-design.md`

## Problem

The quick-prototype story validated that N-BEATS trains and produces a magnitude/confidence prediction on a small recent slice (`--months 6`, ~150 rows after warm-up). `train_model.train()` only proved this end-to-end on that scale: its training loop is full-batch (no `DataLoader`), and it never persists the fitted model — it just logs summary stats and returns arrays.

Issue #6 needs the same pipeline to work against the *complete* cached history — which can be small (`BTC-EUR-1d.csv`, ~2.7k rows) or very large (`BTC-EUR-1m.csv`, 3.18M rows today) — and to save a trained model artifact that a future walk-forward backtest harness ([issue #9](https://github.com/vanesterik/fart/issues/9), not yet built) can load. Because #9 doesn't exist yet, "loadable by the backtest harness" means establishing a stable, self-describing save/load contract now, not integrating with a concrete harness.

## Acceptance Criteria (from issue #6)

- Given the quick-prototype run validated the approach
- When I run full training on the complete dataset
- Then a trained N-BEATS model artifact is saved and loadable by the backtest harness

## Architecture

### Minibatch training

`train()`'s loop switches from full-batch to `torch.utils.data.DataLoader` / `TensorDataset` so it scales to arbitrarily large window counts without holding the whole training set in one forward/backward pass. The train loader shuffles every epoch — standard for N-BEATS, since each window is an independent training example and train/test separation already happened via the existing non-shuffled split, so shuffling afterward introduces no leakage. The eval loader doesn't shuffle, so predictions stay in chronological order. `NBeatsConfig` gains `batch_size: int = 128`.

Per-epoch average loss is now logged via loguru. This wasn't useful at quick-prototype scale (near-instant), but full runs can take substantially longer, and this gives visibility into whether training is progressing.

### GPU support (Apple Silicon MPS)

The pinned `torch` dependency (`pyproject.toml`, `pytorch-cpu` index) already resolves to a plain macOS arm64 wheel with no `+cpu` suffix (confirmed in `uv.lock`) — unlike Linux/Windows, macOS torch wheels bundle MPS support by default, so **no dependency changes are needed**.

New `fart/model/device.py`:

```python
def get_device() -> torch.device:
    """Auto-detect the best available device: MPS if available, else CPU."""
```

Scoped to MPS-or-CPU only — deliberately no `cuda.is_available()` branch, since the `pytorch-cpu` wheel index means CUDA can never be available on Linux/Windows in this project; a CUDA check would be permanently-dead code.

`device` is a new parameter on `train()`, not a field on `NBeatsConfig` — it's a runtime placement concern, not a model hyperparameter, and keeping it out of `config` means it's never accidentally persisted/reconstructed as part of a saved artifact. `device: Optional[torch.device] = None`, where `None` triggers `get_device()`. The model and each batch are moved to `device`; predictions are moved back with `.cpu()` before `.numpy()` (required — MPS tensors can't convert to numpy directly). Before saving, the model itself is moved back to CPU (`model.cpu()`), so `load_model()` always loads a portable, device-agnostic checkpoint (`map_location="cpu"`) regardless of what device trained it; callers `.to(device)` it themselves if they need it on GPU.

CLI gets `--device` (`Optional[str]`, unset = auto-detect), passed through as `torch.device(device)` when set.

**Known limitation:** if a specific op turns out to be unsupported on MPS, torch will raise naturally (no special handling here); `--device cpu` is the escape hatch. This is also only catchable by running on Apple Silicon hardware directly — non-Mac CI (if any exists for this project) will never exercise the MPS code path, since `get_device()` falls back to CPU there.

### Versioning

Filenames get a UTC datetime prefix instead of overwriting on every run. New `fart/utils.py` functions, mirroring the existing `get_candle_filepath`/`get_last_modified_data_file` style:

```python
def get_model_filepath(
    artifacts_dir: Path, market: str, interval: str, timestamp: datetime
) -> Path:
    """artifacts_dir / f'{timestamp:%Y%m%dT%H%M%S%fZ}-{market}-{interval}-nbeats.pt'"""

def get_latest_model_filepath(artifacts_dir: Path, market: str, interval: str) -> Path:
    """Most recent artifact for a market/interval, by filename (not mtime)."""
```

The prefix uses microsecond resolution (`%f`), not just seconds — verified empirically while writing the implementation plan: two `train()` calls in quick succession (e.g. two runs in the same test) can land in the same second, and second-resolution timestamps silently collided, with the second run overwriting the first. Microsecond resolution avoids this for any realistic training run duration.

The prefix (not suffix) position means a plain `ls` sorts artifacts chronologically. `get_latest_model_filepath` globs `*-{market}-{interval}-nbeats.pt` and picks the max by filename — filename rather than filesystem mtime, since the timestamp is embedded in the name and is more robust than mtime against copies/checkouts. It raises `ValueError` on no matches, via an unguarded `max()` — same behavior as the existing sibling `get_last_modified_data_file`, matching that function's established style rather than introducing a different error-handling convention.

This lookup helper exists so a future consumer (the backtest harness, #9) doesn't need to know exact run timestamps — it can always resolve "the latest model for this market/interval."

### Persistence

New `fart/model/nbeats_persistence.py`:

```python
def save_model(model: NBeatsNet, config: NBeatsConfig, path: Path) -> None:
    """Bundle {state_dict, config.model_dump()} into one torch.save file."""

def load_model(path: Path) -> NBeatsNet:
    """Reconstruct NBeatsNet from a checkpoint's bundled config, load
    weights (map_location='cpu'), return in eval mode."""
```

Self-describing: `load_model` doesn't need external metadata to know what architecture to instantiate before loading weights into it. Kept as its own module, matching the existing pattern of small, independently-testable units (`nbeats.py` / `nbeats_dataset.py` / `nbeats_config.py`).

### `train()` signature

```python
def train(
    data_dir: Path,
    market: str,
    interval: str,
    artifacts_dir: Path,
    months: Optional[int] = 6,
    config: Optional[NBeatsConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

`artifacts_dir` is required (no default), matching the existing `data_dir`/`market`/`interval` convention — only the CLI supplies a default (`"artifacts"`, the existing gitignored directory).

### CLI

`fart train` gains three options, following the existing `Annotated[..., typer.Option(...)]` pattern:

```python
months: Annotated[int, typer.Option(help="... Ignored if --full is set.")] = 6,
full: Annotated[bool, typer.Option("--full", help="Train on the complete cached history instead of the recent --months slice.")] = False,
artifacts_dir: Annotated[str, typer.Option(help="Folder to save the trained model artifact to.")] = "artifacts",
device: Annotated[Optional[str], typer.Option(help="Torch device to train on ('cpu', 'mps'). Auto-detected if not set.")] = None,
```

Passed through as `months=None if full else months`, `artifacts_dir=Path(artifacts_dir)`, `device=torch.device(device) if device else None`.

## Data Flow

```
fart train --full
  → train_model.train(..., months=None, artifacts_dir=Path("artifacts"), device=None)
      → prepare_training_data(..., months=None)        # UNCHANGED, already supports None
      → build_return_windows(...)                       # UNCHANGED
      → device = get_device()                            # MPS if available, else CPU
      → model = NBeatsNet(config).to(device)
      → DataLoader(TensorDataset(X_train_windows, y_train_windows), batch_size, shuffle=True)
      → per epoch: iterate minibatches (each moved to device), GaussianNLLLoss,
        log avg epoch loss
      → eval via DataLoader(shuffle=False) over test windows (each batch moved to
        device, predictions moved back to CPU), concatenate
      → model.cpu()
      → timestamp = datetime.now(timezone.utc)
      → save_model(model, config, get_model_filepath(artifacts_dir, market, interval, timestamp))
      → log saved path and device used
      → return (magnitudes, confidences)                 # UNCHANGED return contract
```

## Error Handling

No new failure modes beyond what's already documented (`FileNotFoundError` from `prepare_training_data` on a missing CSV; `ValueError` from `build_return_windows` on an undersized slice). `save_model` creates `artifacts_dir` if missing (`mkdir(parents=True, exist_ok=True)`). `get_latest_model_filepath` raises `ValueError` when no artifact matches — not exercised by `train()` itself (it only ever writes), only by future consumers of the lookup helper.

## Testing

- `tests/model/test_nbeats_persistence.py` — save/load round trip: save a small model, reload it, assert config equality and identical output for the same input.
- `tests/model/test_device.py` — `get_device()` with `torch.backends.mps.is_available` mocked both `True` and `False`.
- `tests/utils/test_get_model_filepath.py` — deterministic path given a fixed timestamp, matching the one-file-per-function convention already used for `get_candle_filepath`.
- `tests/utils/test_get_latest_model_filepath.py` — picks the max by filename among multiple candidates; raises `ValueError` when none match.
- `tests/model/test_train_model.py`:
  - Existing tests updated to pass `artifacts_dir=tmp_path` and `device=torch.device("cpu")` explicitly — required now (`artifacts_dir` has no default) and necessary to keep test behavior deterministic and independent of whatever machine runs them, rather than silently depending on MPS availability.
  - New: minibatching works with `batch_size` smaller than the window count (finite loss, gradients flow).
  - New: two consecutive `train()` calls produce two distinct artifact files under `tmp_path` matching the naming pattern (proves versioning, not overwrite).
  - New: the saved artifact is loadable via `load_model()` and reproduces the same magnitude/confidence output as the original in-memory model on the same test windows (an end-to-end round trip through the full `train()` pipeline, not just the persistence unit test).

## Out of Scope

- Actual integration with the backtest harness (#9) — only the save/load + latest-lookup contract.
- CUDA support — project-wide `pytorch-cpu` wheel pin means CUDA is unavailable on Linux/Windows; MPS/CPU only.
- Artifact cleanup/retention policy — versioned artifacts accumulate indefinitely in `artifacts/`; no pruning.
- Hyperparameter tuning of `NBeatsConfig` defaults for full-scale data quality — `epochs=50`, `hidden_width=64`, etc. stay as-is; `batch_size=128` is a reasonable starting default, not a tuned result.
- Streaming/lazy windowing — `build_return_windows` still builds all windows eagerly; even at 3.18M rows (`BTC-EUR-1m`) this is ~400MB of float32, which is fine in memory.

## Open Questions

None — architecture, versioning scheme, device selection, and the save/load contract were all confirmed during design.
