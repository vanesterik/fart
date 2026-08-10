# N-BEATS Ensemble Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research notebook that trains a 5-member deep ensemble of `NBeatsNet` point predictors (no self-reported log-variance), derives a confidence score from cross-member prediction disagreement, and compares its correlation with actual error against the existing self-reported-confidence approach diagnosed in `notebooks/2.0-kve-nbeats-confidence-calibration.ipynb`.

**Architecture:** A new notebook, `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`, loads the same full BTC-EUR/1d candle history and window split as notebook 2.0, then trains 5 independently-seeded `NBeatsNet(NBeatsConfig())` instances with plain MSE loss on the `mu` output only (the model's `log_sigma` output is computed but never appears in the loss, so it's simply discarded). Per test window, the 5 members' predictions are aggregated into a mean (`mu_ensemble`) and a std (`sigma_ensemble`); `confidence = 1/(1+sigma_ensemble)` reuses the exact squashing formula the current self-reported-confidence model already applies, so the existing `plot_confidence_calibration()` helper can be called unmodified for a like-for-like comparison.

**Tech Stack:** Python 3.14, PyTorch (CPU/MPS via `fart.model.device.get_device`), Polars, NumPy, Jupyter, matplotlib (via `plot_confidence_calibration`), uv.

## Global Constraints

- No changes to `src/fart` — this work is purely additive (one new notebook), per `docs/specs/2026-08-10-nbeats-ensemble-confidence-design.md` (spec decision).
- New file only: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`, following the existing `N.0-kve-<topic>` numbering (`1.0`, `2.0`, `3.0` already exist).
- Data must be loaded with `months=None` explicitly. `prepare_training_data`'s default is `months=6` (`src/fart/model/train_model.py:30`); the spec requires the full ~2706-candle history to match notebook 2.0's stated setup, so `months` must be passed explicitly rather than omitted.
- Confidence formula is `confidence = 1 / (1 + sigma_ensemble)` — the same formula the current model uses for `1/(1+exp(log_sigma))`, just fed an empirical ensemble std instead of a learned one (spec decision, keeps scale comparable to notebook 2.0).
- Exactly 5 ensemble members, all using the same `NBeatsConfig()` defaults (no hyperparameter variation across members) — diversity comes only from `torch.manual_seed(seed)` for `seed in range(5)` before each member's construction, matching the standard deep-ensembles recipe (spec decision).
- Reuse `src/fart/visualization/confidence_calibration.py::plot_confidence_calibration(confidence, error)` unmodified for the final plot.
- No new automated tests — notebook-only research work, consistent with how notebooks 2.0 and 3.0 were handled (`tests/` doesn't cover notebooks in this repo).
- `lefthook`'s `clean-jupyter` pre-commit hook strips notebook outputs automatically on commit — no manual output-stripping step is needed before committing.
- Work happens on the already-checked-out `nbeats-ensemble-confidence` branch.
- Never reference "superpowers" in file paths (CLAUDE.md).

---

## Task 1: Notebook shell — imports, setup, and data preparation

**Files:**
- Create: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`

**Interfaces:**
- Produces: notebook variables `config: NBeatsConfig`, `lookback: int`, `X_train_windows: torch.Tensor` (shape `(n_train_windows, lookback)`), `y_train_windows: torch.Tensor` (shape `(n_train_windows,)`), `X_test_windows: torch.Tensor` (shape `(n_test_windows, lookback)`), `y_test_windows: torch.Tensor` (shape `(n_test_windows,)`) — consumed by Task 2's training loop and Task 3's aggregation.

- [ ] **Step 1: Create the notebook file with an empty cell list**

Write this exact JSON to `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` (metadata copied from `notebooks/2.0-kve-nbeats-confidence-calibration.ipynb` for a consistent kernel):

```json
{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "fart (3.14.6)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

- [ ] **Step 2: Read the notebook**

Use the Read tool on `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` (required before any `NotebookEdit` call).

- [ ] **Step 3: Insert the setup/imports cell**

Use `NotebookEdit` with `notebook_path=notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`, `edit_mode="insert"`, `cell_type="code"` (omit `cell_id` — inserts at the beginning), `new_source`:

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from fart.model.device import get_device
from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.train_model import prepare_training_data
from fart.utils import get_project_root
from fart.visualization.confidence_calibration import plot_confidence_calibration
from fart.visualization.plot_styles import apply_plot_styles

apply_plot_styles()
```

- [ ] **Step 4: Read the notebook again to find the new cell's id**

Use Read on the notebook. Note the `id` shown in `<cell id="...">` for the cell just inserted — you'll need it as `cell_id` in the next step.

- [ ] **Step 5: Insert the data-preparation cell after the setup cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, `cell_id=<id from Step 4>`, `new_source`:

```python
assets_dir = get_project_root() / "assets"
market, interval = "BTC-EUR", "1d"

config = NBeatsConfig()
lookback = config.lookback

X_train, X_test, y_train, y_test = prepare_training_data(
    data_dir=assets_dir,
    market=market,
    interval=interval,
    months=None,
)

n_train = y_train.shape[0]
close_prices = pl.concat([y_train, y_test])
X_all, y_all = build_return_windows(close_prices, lookback)

n_train_windows = max(0, n_train - lookback - 1)
X_train_windows, y_train_windows = X_all[:n_train_windows], y_all[:n_train_windows]
X_test_windows, y_test_windows = X_all[n_train_windows:], y_all[n_train_windows:]

len(close_prices), X_train_windows.shape, X_test_windows.shape
```

- [ ] **Step 6: Execute the notebook so far and verify shapes**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0, no exceptions. Then Read the notebook and check the last cell's output: `len(close_prices)` should be close to 2706 (the full `BTC-EUR-1d.csv` row count, confirming `months=None` pulled the full history, not a 6-month slice), and `X_train_windows.shape[0] + X_test_windows.shape[0]` should be `len(close_prices) - lookback - 1`.

- [ ] **Step 7: Commit**

```bash
git add notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
git commit -m "$(cat <<'EOF'
feat: scaffold ensemble-confidence notebook with data prep

Loads the full BTC-EUR/1d history (months=None, matching notebook
2.0's setup) and builds the same sliding-window train/test split,
as a base for training the 5-member ensemble in the next commit.
EOF
)"
```

---

## Task 2: Train the 5-member ensemble

**Files:**
- Modify: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`

**Interfaces:**
- Consumes: `config: NBeatsConfig`, `X_train_windows`, `y_train_windows`, `X_test_windows` from Task 1.
- Produces: notebook variable `predictions: torch.Tensor`, shape `(5, n_test_windows)`, CPU tensor — each row is one ensemble member's `mu` prediction on every test window. Consumed by Task 3's aggregation.

- [ ] **Step 1: Read the notebook**

Read `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` to get current cell ids.

- [ ] **Step 2: Insert the ensemble training cell after the data-prep cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, `cell_id=<id of the data-prep cell from Task 1>`, `new_source`:

```python
device = get_device()
num_members = 5

member_predictions: list[torch.Tensor] = []

for seed in range(num_members):
    torch.manual_seed(seed)
    model = NBeatsNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = DataLoader(
        TensorDataset(X_train_windows, y_train_windows),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    epoch_loss = 0.0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mu, _log_sigma = model(X_batch).unbind(-1)
            loss = torch.nn.functional.mse_loss(mu, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.shape[0]

    model.eval()
    with torch.no_grad():
        mu_test, _log_sigma_test = model(X_test_windows.to(device)).unbind(-1)
    member_predictions.append(mu_test.cpu())

    print(f"member {seed}: final epoch loss = {epoch_loss / len(X_train_windows):.6f}")

predictions = torch.stack(member_predictions)
predictions.shape
```

- [ ] **Step 3: Execute and verify**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0. Read the notebook and check this cell's output: 5 printed `member N: final epoch loss = ...` lines, each a finite, non-NaN number generally decreasing from member to member's own epoch 1 vs epoch 50 trend (not required to compare across members). Final output `predictions.shape` should read `torch.Size([5, n_test_windows])`, where `n_test_windows` matches `X_test_windows.shape[0]` from Task 1.

- [ ] **Step 4: Commit**

```bash
git add notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
git commit -m "$(cat <<'EOF'
feat: train 5-member N-BEATS ensemble with plain MSE loss

Each member is a freshly-seeded NBeatsNet trained on mu only (MSE);
log_sigma is computed but unused, so those weights stay at random
init. Collects all 5 members' test-window predictions for the
disagreement-based confidence metric in the next commit.
EOF
)"
```

---

## Task 3: Aggregate ensemble predictions into error and confidence

**Files:**
- Modify: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`

**Interfaces:**
- Consumes: `predictions: torch.Tensor` (shape `(5, n_test_windows)`) from Task 2, `y_test_windows: torch.Tensor` from Task 1.
- Produces: notebook variables `error: np.ndarray` (shape `(n_test_windows,)`, float64), `confidence: np.ndarray` (shape `(n_test_windows,)`, float64, values in `(0, 1)`) — consumed by Task 4's plot call.

- [ ] **Step 1: Read the notebook**

Read `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` to get current cell ids.

- [ ] **Step 2: Insert the aggregation cell after the training cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, `cell_id=<id of the training cell from Task 2>`, `new_source`:

```python
mu_ensemble = predictions.mean(dim=0).numpy()
sigma_ensemble = predictions.std(dim=0).numpy()

error = np.abs(y_test_windows.numpy() - mu_ensemble)
confidence = 1 / (1 + sigma_ensemble)

print(f"confidence range: [{confidence.min():.5f}, {confidence.max():.5f}]")
print(f"error range: [{error.min():.6f}, {error.max():.6f}]")
```

- [ ] **Step 3: Execute and verify**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0. Read the notebook and check the printed ranges: `confidence` values must fall strictly between 0 and 1 (guaranteed by the `1/(1+x)` formula for `x >= 0`, since `sigma_ensemble` is a std and can't be negative), and `error` values must be `>= 0`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
git commit -m "$(cat <<'EOF'
feat: derive confidence from ensemble prediction disagreement

confidence = 1/(1+sigma_ensemble) reuses the same squashing formula
the self-reported-confidence model already applies, so the result is
directly comparable to notebook 2.0's plot on the same scale.
EOF
)"
```

---

## Task 4: Plot and compare against notebook 2.0

**Files:**
- Modify: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`

**Interfaces:**
- Consumes: `confidence: np.ndarray`, `error: np.ndarray` from Task 3.
- Produces: rendered plot (no new notebook variables consumed elsewhere).

- [ ] **Step 1: Read the notebook**

Read `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` to get current cell ids.

- [ ] **Step 2: Insert the plot cell after the aggregation cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, `cell_id=<id of the aggregation cell from Task 3>`, `new_source`:

```python
plot_confidence_calibration(confidence, error)
```

- [ ] **Step 3: Execute and read the resulting Pearson r**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0, a figure is produced. Read the notebook; the figure's `suptitle` (rendered in the cell's image output, visible via the Read tool) reads `Confidence vs. actual error  (n=<n_test_windows>, Pearson r=<value>)`. Note the exact `n` and `r` values printed — you'll use them in the next step.

- [ ] **Step 4: Read the notebook to get the plot cell's id**

Read `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` to get the plot cell's id for the next insert.

- [ ] **Step 5: Insert a markdown comparison cell after the plot cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="markdown"`, `cell_id=<id of the plot cell from Step 4>`, `new_source` — replace `<n>` and `<r>` with the exact values read in Step 3, and replace `<conclusion>` with one sentence stating whether `|r|` increased relative to notebook 2.0's `0.043` and whether the confidence spread (`confidence.min()`/`confidence.max()` from Task 3's printed output) widened relative to notebook 2.0's `[0.965, 0.976]` band:

```markdown
## Comparison to notebook 2.0

| | Self-reported confidence (notebook 2.0) | Ensemble disagreement (this notebook) |
|---|---|---|
| n (test windows) | 535 | <n> |
| Pearson r | 0.043 | <r> |
| Confidence range | [0.965, 0.976] | see printed range above |

<conclusion>
```

- [ ] **Step 6: Execute once more to confirm the markdown cell renders cleanly**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0, no exceptions (markdown cells aren't executed but this confirms the notebook as a whole is still valid and re-runs end to end).

- [ ] **Step 7: Commit**

```bash
git add notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
git commit -m "$(cat <<'EOF'
feat: plot ensemble confidence calibration and compare to notebook 2.0

Reuses plot_confidence_calibration() unmodified for a like-for-like
comparison against the self-reported-confidence collapse diagnosed
in notebook 2.0 (r=0.043, band 0.965-0.976).
EOF
)"
```

---

## Task 5: Full clean re-run and final verification

**Files:**
- Modify: `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` (outputs only, via re-execution)

**Interfaces:**
- Consumes: the complete notebook from Tasks 1-4.
- Produces: nothing new — this is an end-to-end sanity check before considering the notebook done.

- [ ] **Step 1: Clear all outputs and re-run from scratch**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ClearOutputPreprocessor.enabled=True notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
```

Expected: exits 0. This clears any stale outputs from incremental development in Tasks 1-4 and re-runs the whole notebook top to bottom in one pass, catching any cell-ordering or stale-variable bugs that incremental per-task execution could have masked.

- [ ] **Step 2: Read the notebook end to end**

Read `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` in full. Confirm: no error outputs in any cell, the data-prep cell's output still shows the full ~2706-row history, the training cell shows 5 finite member losses, the aggregation cell's confidence/error ranges are still sane (confidence in `(0,1)`, error `>= 0`), the plot renders, and the final markdown comparison cell's `<n>`/`<r>`/`<conclusion>` placeholders were all filled with real values (not literal `<n>`/`<r>`/`<conclusion>` text) in Task 4.

- [ ] **Step 3: Commit if the re-run changed anything**

```bash
git status
```

If `git status` shows changes to `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb` (e.g. outputs differ from the clean re-run), commit them:

```bash
git add notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb
git commit -m "$(cat <<'EOF'
chore: re-run ensemble-confidence notebook clean end to end
EOF
)"
```

If `git status` shows no changes, no commit is needed — the notebook was already consistent (note: the pre-commit `clean-jupyter` hook strips outputs on every commit regardless, so a truly clean diff here is expected and fine).
