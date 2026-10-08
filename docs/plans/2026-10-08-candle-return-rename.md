# Candle Return Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the code's "magnitude" (the signed per-candle percent change) to "candle return", so its name says what it is, with no change in behaviour (story #74).

**Architecture:** A mechanical rename following the mapping in the epic C design: identifiers and modules via ordered word-boundary replacements, then a manual pass over prose, chart labels and markdown cells. Task 1 covers `src` and `tests`; Task 2 the notebooks and docs. The test suite and a re-execution of all three notebooks prove that behaviour is unchanged.

**Tech Stack:** perl (word-boundary replacements), git mv, pytest, jupyter nbconvert, Task.

**Spec:** `docs/specs/2026-10-08-forecast-tools-design.md` (Decisions: "It's called a return, not a magnitude", with the rename table)

## Global Constraints

- The mapping, from the design:

  | Now | After |
  |---|---|
  | `MAGNITUDE = "Magnitude"` | `CANDLE_RETURN = "Return"` |
  | `features/calculate_magnitude.py::calculate_magnitude` | `features/calculate_candle_returns.py::calculate_candle_returns` |
  | `calculate_trade_returns(magnitudes, predicted_magnitudes, ...)` | `calculate_trade_returns(candle_returns, predicted_candle_returns, ...)` |
  | `visualization/magnitude.py::plot_magnitude` | `visualization/candle_returns.py::plot_candle_returns` |
  | `tests/features/test_calculate_magnitude.py` | `tests/features/test_calculate_candle_returns.py` |

- "Candle return", not plain "return": `calculate_trade_returns` already uses "returns" for per-trade results.
- Prose that uses "magnitude" in its ordinary sense (size without sign) stays: `visualization/diverging_bar_chart.py` ("magnitude and direction are readable at a glance") and `visualization/predicted_vs_actual_scatter.py`'s docstring ("…the model learned magnitude or has collapsed toward a near-constant prediction").
- Behaviour doesn't change: the same tests pass with only names changed. Before the rename, all three notebooks execute cleanly (`1.0` in 8 s, `2.0` in 11 s, `2.1` in 90 s, measured while writing this plan), and they must afterwards too.
- The cached CSVs have no `Magnitude` column (it's computed on load), so no data file changes.
- The historical specs and plans in `docs/` keep the old name.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **A keyword argument call site:** someone calls `calculate_trade_returns(magnitudes=..., predicted_magnitudes=...)`. Expect every call site in `src`, `tests` and the notebooks to use the new names. The `git grep` checks in both tasks cover it, and the notebook execution in Task 2 catches any missed keyword at runtime.
- **A notebook imports the old module path** (`fartt.features.calculate_magnitude`, `fartt.visualization.magnitude`). Expect it to fail loudly at execution. Task 2 executes all three.
- **A chart label still says "Magnitude".** Expect "Return" or "Candle return" wherever a label names the quantity. Covered by Task 1's manual label pass and Task 2's grep over the notebooks.
- **The DataFrame column name changes from "Magnitude" to "Return".** Code that looks the column up by string instead of through the constant would break. Expect none: the grep for `"Magnitude"` in Task 1 Step 3 and the notebook execution cover it.
- **A test passes only because it was renamed along with a bug.** Expect only names to change in the tests. The diff of `tests/` must show renames only; Task 1 Step 5 checks it with `git diff --word-diff`.

---

## Task 1: Rename in `src` and `tests`

**Files** (12, over the usual ten: the rename can't be split without leaving imports broken in between):
- Rename: `src/fartt/features/calculate_magnitude.py` → `calculate_candle_returns.py`
- Rename: `src/fartt/visualization/magnitude.py` → `candle_returns.py`
- Rename: `tests/features/test_calculate_magnitude.py` → `test_calculate_candle_returns.py`
- Modify: `src/fartt/constants.py`, `src/fartt/features/calculate_trade_returns.py`, `src/fartt/features/align_predicted_signal.py`, `src/fartt/model/prepare_datasets.py`, `src/fartt/visualization/predicted_vs_actual_scatter.py`
- Modify: `tests/features/test_calculate_trade_returns.py`, `tests/features/test_align_predicted_signal.py`, `tests/model/test_prepare_datasets.py`

- [ ] **Step 1: Move the three files**

```bash
git mv src/fartt/features/calculate_magnitude.py src/fartt/features/calculate_candle_returns.py
git mv src/fartt/visualization/magnitude.py src/fartt/visualization/candle_returns.py
git mv tests/features/test_calculate_magnitude.py tests/features/test_calculate_candle_returns.py
```

- [ ] **Step 2: Apply the identifier replacements, in this order**

The order matters: the longer names go first, so `predicted_magnitudes` isn't half-replaced by the `magnitudes` rule.

```bash
FILES=$(git grep -liE "magnitude" -- src tests)
perl -pi -e '
  s/\bpredicted_magnitudes\b/predicted_candle_returns/g;
  s/\bcalculate_magnitude\b/calculate_candle_returns/g;
  s/\bplot_magnitude\b/plot_candle_returns/g;
  s/\bvisualization\.magnitude\b/visualization.candle_returns/g;
  s/\bMAGNITUDE\b/CANDLE_RETURN/g;
  s/\bmagnitudes\b/candle_returns/g;
' $FILES
```

- [ ] **Step 3: Fix the constant's value, the variables and the prose by hand**

- `src/fartt/constants.py`: `CANDLE_RETURN = "Return"`. Keep it in alphabetical order within `# Labels`: it moves from the `M`s to the `C`s.
- `src/fartt/features/calculate_candle_returns.py`: the docstring says "Calculate each candle's return: the signed percent change of Close from the previous candle, as a fraction. Sign carries direction (positive = price increase, negative = price decrease)." and "`df` with a `Return` column added."
- `src/fartt/features/calculate_trade_returns.py`:
  - in the loop, rename the local `magnitude = values[i]` to `candle_return`, along with its two uses;
  - in the docstring, "signed percent-change magnitudes" becomes "candle returns (signed percent changes, as fractions)", `magnitude[i + 1]` becomes `candle_returns[i + 1]`, "compounded magnitudes" becomes "compounded candle returns", and "`calculate_candle_returns`'s `Magnitude` column" becomes "…'s `Return` column".
- `src/fartt/features/align_predicted_signal.py`: the docstring's `magnitudes[i + 1]` is already handled by Step 2. Check that it reads `candle_returns[i + 1]`.
- `src/fartt/model/prepare_datasets.py`: "calculating the magnitude of the target column" becomes "calculating each candle's return (`calculate_candle_returns`)".
- `src/fartt/visualization/candle_returns.py`:
  - the docstring "Plot the Magnitude column" becomes "Plot the candle return column", and "`Datetime` and `Magnitude` columns" becomes "`Datetime` and `Return` columns";
  - the local `magnitude = ...` becomes `candle_returns`;
  - the title "Magnitude of Close Price Movements" becomes "Candle Returns (Signed Change of Close)";
  - the y label `CANDLE_RETURN` (from Step 2) now shows "Return", which is correct.
- `src/fartt/visualization/predicted_vs_actual_scatter.py`: the labels become "Actual Return", "Predicted Return" and "Predicted vs. Actual Return (Test Set)". The docstring keeps "magnitude" in its ordinary sense.
- `tests/features/test_calculate_candle_returns.py`: rename the test functions from `test_calculate_magnitude_*` to `test_calculate_candle_returns_*`, if Step 2 didn't already.

Then check what's left:

Run: `git grep -niE "magnitude" -- src tests`
Expected: only `diverging_bar_chart.py` ("magnitude and direction") and `predicted_vs_actual_scatter.py`'s docstring ("learned magnitude or has collapsed"), both in the ordinary sense.

Run: `git grep -n '"Magnitude"' -- src tests notebooks`
Expected: matches only in the notebooks, which Task 2 handles.

- [ ] **Step 4: Run the checks**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `136 passed, 3 deselected`, the same count as before, since only names changed.

- [ ] **Step 5: Confirm the tests changed in names only**

Run: `git diff --word-diff=plain -M -- tests | grep -E "^\[-|\{\+" | grep -viE "magnitude|candle_return|CANDLE_RETURN" | head`
Expected: no output. Every changed word in `tests/` is part of the rename.

- [ ] **Step 6: Commit**

```bash
git add -A src tests
git commit -m "refactor: rename magnitude to candle return in the code and tests"
```

---

## Task 2: Rename in the notebooks and docs

**Files:**
- Modify: `notebooks/1.0-kve-data-exploration.ipynb`, `notebooks/2.0-kve-data-analysis-mlp.ipynb`, `notebooks/2.1-kve-data-analysis-cnn.ipynb`
- Modify: `CLAUDE.md`, `README.md`

- [ ] **Step 1: Apply the same replacements to the notebooks**

```bash
perl -pi -e '
  s/\bpredicted_magnitudes\b/predicted_candle_returns/g;
  s/\bcalculate_magnitude\b/calculate_candle_returns/g;
  s/\bplot_magnitude\b/plot_candle_returns/g;
  s/\bvisualization\.magnitude\b/visualization.candle_returns/g;
  s/\bMAGNITUDE\b/CANDLE_RETURN/g;
  s/\bmagnitudes\b/candle_returns/g;
' notebooks/*.ipynb
```

Then read every remaining `magnitude` (case-insensitive) in the three notebooks:

Run: `grep -noiE "[^\"]{0,40}magnitude[^\"]{0,40}" notebooks/*.ipynb`

Rewrite those in markdown cells and comments that name the quantity (e.g. "the Magnitude column", "magnitude of each candle") as "candle return". Leave any that mean size without sign.

Run: `python3 -c "import json,glob;[json.load(open(f)) for f in glob.glob('notebooks/*.ipynb')];print('valid')"`
Expected: `valid`.

- [ ] **Step 2: Execute all three notebooks**

Run:

```bash
mkdir -p /tmp/fartt-nb-check
for nb in notebooks/*.ipynb; do
  uv run jupyter nbconvert --to notebook --execute "$nb" --output-dir /tmp/fartt-nb-check --ExecutePreprocessor.timeout=900 >/dev/null 2>&1
  echo "$(basename "$nb"): exit $?"
done
rm -rf /tmp/fartt-nb-check
```

Expected: `exit 0` for all three, the same as before the rename.

- [ ] **Step 3: Update `CLAUDE.md` and the README**

- `CLAUDE.md`:
  - "computes `Magnitude` (signed percent-change)" becomes "computes the candle return (`calculate_candle_returns`, the signed percent change as a fraction, in the `Return` column)", wherever it appears: the pipeline bullet in Current state, and the Architecture `fartt/model/` bullet;
  - the Backtesting helpers bullet says `calculate_candle_returns.py` and `predicted_candle_returns`;
  - the `fartt/features/` Architecture bullet says `calculate_candle_returns.py` and "over a candle-return series";
  - the `fartt/visualization/` bullet says `candle_returns.py`.
- `README.md`:
  - the tree says `calculate_candle_returns.py`, and the `prepare_datasets.py` line reads "Loads candles, computes candle returns, builds lag windows + split.";
  - the Usage paragraph on the notebooks says "computes the target (the candle return, the signed percent change as a fraction)".

Run: `grep -niE "magnitude" CLAUDE.md README.md`
Expected: no matches, or only the ordinary sense.

- [ ] **Step 4: Check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `136 passed, 3 deselected`.

```bash
git add notebooks CLAUDE.md README.md
git commit -m "refactor: rename magnitude to candle return in the notebooks and docs"
git push
```
