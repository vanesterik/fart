# Remove Superseded Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the code, dependencies and documentation the MCP revision superseded, so nobody (human or agent) mistakes the old platform scaffolding for the current design (story #53, the last story of epic #44).

**Architecture:** No new code. `src/fartt/core/` and everything only it needs go: three dependencies, a type stub, 14 constants and editor spell-check words. The unused `.env` loading goes too. `CLAUDE.md` and the README are then rewritten around what exists now: the exchange layer, candle cache, MCP server, the model pipeline epic B builds on, and the PRD's epics.

**Tech Stack:** uv, Task, pytest, pyright, ruff, Markdown and Mermaid.

**Spec:** `docs/specs/2026-10-07-server-foundation-design.md` (section "4. Removal and documentation"), plus the `.env` scenario issue #53 gained from the #51 review.

## Global Constraints

- Delete only what nothing else uses. Each removal is backed by a search over `src`, `tests` and `notebooks` (tracked files, `git grep`), and `task check` stays green.
- `.env` handling goes because nothing reads environment variables since #51. API keys come back under exchange-neutral names in epics D and F, and their `.env.example` comes back with them.
- `.gitignore` keeps ignoring `.env`, so a key file someone creates later can't be committed by accident.
- The historical specs and plans in `docs/specs/` and `docs/plans/` stay as they are: they record past work, and ruff already excludes `docs/`.
- The documentation describes what exists now and points to the PRD (`docs/product/mcp-trading-agent-prd.md`) and its epics for what's planned. It drops the Part A / Part B framing.
- Never reference "superpowers" in code, file paths or directory structure.

## Review Focus

- **Something outside `core/` still uses a removed name** (a notebook using `EUR` or a colour constant, or a test importing `fartt.core`): expect `task check` and the `git grep` steps to catch it before the commit. Notebooks aren't type-checked, hence the explicit search over `notebooks`.
- **A fresh environment after the dependency removals:** expect `uv sync` to remove `python-bitvavo-api`, `rich` (unless another package needs it transitively), `babel` and `python-dotenv`, and the suite to still pass. Checked in Task 1 Step 4 and Task 2 Step 3.
- **The documentation no longer matches the code:** expect every module, command and file the rewritten docs name to exist. Checked in Task 3 Step 4 by extracting each path and command from the two files and testing it.

---

## Task 1: Remove `core/` and what only it used

**Files:**
- Delete: `src/fartt/core/` (`broker.py`, `dashboard.py`, `exchange.py`)
- Delete: `typings/python_bitvavo_api/`
- Modify: `src/fartt/constants.py` (remove 14 dashboard-only constants)
- Modify: `pyproject.toml`, `uv.lock` (`uv remove python-bitvavo-api rich babel`)
- Modify: `.vscode/settings.json` (spell-check words only the old code used)

- [ ] **Step 1: Confirm nothing outside `core/` imports it or its dependencies**

Run: `git grep -nE "fartt\.core|from fartt import core|python_bitvavo_api|^\s*(from|import) rich|babel" -- src tests notebooks`
Expected: matches only in `src/fartt/core/dashboard.py` and `src/fartt/core/exchange.py`.

- [ ] **Step 2: Delete the package and its stub**

Run: `git rm -rq src/fartt/core typings/python_bitvavo_api`

- [ ] **Step 3: Remove the dashboard-only constants**

These 14 names are used only by `core/dashboard.py` (checked with `git grep -lw <name> -- src tests notebooks`):
`DOVE_GREY`, `GREEN`, `RED`, `BALANCE`, `CHANGE`, `LAST_UPDATE`, `NOT_AVAILABLE`, `PRICE`, `PROFIT_LOSS`, `THIS_MONTH`, `THIS_WEEK`, `TODAY`, `TOTAL`, `YEAR_TO_DATE`.

Delete their lines from `src/fartt/constants.py`; keep the `# Colors` and `# Labels` sections and the order of what remains. Then confirm none is referenced anywhere:

Run: `for n in DOVE_GREY GREEN RED BALANCE CHANGE LAST_UPDATE NOT_AVAILABLE PRICE PROFIT_LOSS THIS_MONTH THIS_WEEK TODAY TOTAL YEAR_TO_DATE; do git grep -qw "$n" -- src tests notebooks && echo "still used: $n"; done; echo done`
Expected: only `done`.

- [ ] **Step 4: Drop the dependencies**

Run: `uv remove python-bitvavo-api rich babel`
Expected: all three leave `[project] dependencies` in `pyproject.toml`; `uv.lock` updates.

Run: `uv tree --invert --package rich --quiet 2>/dev/null | head -3`
Expected: `rich` either gone or only pulled in by another package (that's fine; it's no longer a direct dependency).

- [ ] **Step 5: Drop the old spell-check words**

In `.vscode/settings.json`, remove `"APIKEY"`, `"APISECRET"` and `"Renderable"` from `cSpell.words` (used only by the Bitvavo client keys and the rich dashboard). Keep the others.

- [ ] **Step 6: Check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `136 passed, 1 deselected` (no tests covered `core/`).

Run: `ls src/fartt/core 2>&1; git grep -nE "fartt\.core|python_bitvavo" -- src tests notebooks pyproject.toml; echo done`
Expected: `ls: src/fartt/core: No such file or directory`, then `done`.

```bash
git add -A src/fartt/core typings/python_bitvavo_api src/fartt/constants.py pyproject.toml uv.lock .vscode/settings.json
git commit -m "chore: remove the superseded core package and its dependencies"
```

---

## Task 2: Stop loading credentials nothing reads

**Files:**
- Modify: `src/fartt/cli.py` (drop `load_dotenv`)
- Modify: `pyproject.toml`, `uv.lock` (`uv remove python-dotenv`)
- Delete: `.env.example`

- [ ] **Step 1: Confirm nothing reads environment variables**

Run: `git grep -nE "getenv|os\.environ|dotenv" -- src tests notebooks`
Expected: only the `dotenv` import and `load_dotenv(find_dotenv())` call in `src/fartt/cli.py`.

- [ ] **Step 2: Remove the loading and the dependency**

In `src/fartt/cli.py`, delete `from dotenv import find_dotenv, load_dotenv` and the `load_dotenv(find_dotenv())` line in `main()`.

Run: `uv remove python-dotenv && git rm -q .env.example`

- [ ] **Step 3: Check and commit**

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `136 passed, 1 deselected`.

Run: `uv run fartt download --assets-dir /tmp/fartt-env-check --interval 1d 2>&1 | tail -1; rm -rf /tmp/fartt-env-check`
Expected: a completed download (about 2,770 candles): nothing needed the environment.

Run: `git check-ignore -v .env`
Expected: `.gitignore` still ignores `.env`.

```bash
git add src/fartt/cli.py pyproject.toml uv.lock .env.example
git commit -m "chore: stop loading credentials that nothing reads"
```

---

## Task 3: Rewrite `CLAUDE.md` and the README around the MCP architecture

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Rewrite `CLAUDE.md`'s overview and current state**

Replace everything from `## Project overview` up to `## Commands` with:

- **Project overview:** Fartt is an MCP server that gives an agent (Claude Code is the client) the tools to trade BTC, backed by a forecast model. The full plan is the PRD, `docs/product/mcp-trading-agent-prd.md`, delivered in epics on the board: A Server Foundation (#44, done), C Forecast Tools (#45), D Paper Trading & Risk (#46), B Model Selection (#47), E Unattended Operation (#48), F Live Trading (#49), in that order. Keep the "solo research project" and cookiecutter-layout sentences.
- **Current state:** what exists and works:
  - the exchange layer (ccxt, Bitvavo by default);
  - the candle cache;
  - the MCP server with `get_candles`, started by `fartt serve` from `.mcp.json`;
  - `fartt download`;
  - the model pipeline, from `prepare_datasets` through the builders, `train_model`, `evaluate_model` and `persist_model`, exercised from the `2.0`/`2.1` notebooks. Keep the existing detail on the pipeline: the split, the builders and configs, no cross-validation, `loss_history`, and `persist_model` saving the whole module.

  And what doesn't exist yet: forecast tools, paper trading and risk controls, live orders, and `predict_model.py` (still an empty stub, the place epic C's forecaster lands). Keep the notes that `device.py::get_device()` is unused, and that the five-architecture screen (MLP and CNN built; GRU, N-BEATS rebuilt fresh, transformer, plus foundation models and naive baselines) now belongs to epic B. Keep the N-BEATS history note. Keep the note on `calculate_magnitude.py`/`calculate_trade_returns.py` replacing `TradeStrategy`.
- Remove every mention of `core/`, `broker.py`, the dashboard, Part A/Part B, and the deleted Part A/Part B PRDs.

- [ ] **Step 2: Fix `CLAUDE.md`'s Architecture section**

- Delete the `fartt/core/exchange.py` and `fartt/core/dashboard.py` bullets.
- `fartt/constants.py`: "centralized column names, indicator labels and a small colour palette" (no "UI labels").
- `fartt/model/`: replace "Part A's regression pipeline" with "the regression pipeline epic B (Model Selection) builds on", and "This replaced an earlier hand-rolled N-BEATS prototype (see "Current state" above)" stays.

- [ ] **Step 3: Rewrite the README**

- **Intro (top paragraph):** Fartt is an MCP server that gives an AI agent the tools to trade BTC (market data now; forecasts, portfolio, risk controls and orders as the epics land), with Claude Code as the agent and the operator's interface. Keep the banner, the name's expansion, Motivation and Name.
- **Project Status:** replace the Part A/Part B bullets with:
  - a short paragraph on the shift: from a self-built trading platform to an MCP server operated by an agent, with a link to the PRD;
  - the epic list, in build order: done (A), next (C), later (D, B, E, F). One line each, linked to its issue;
  - one sentence on what works today: `fartt download`, `fartt serve` with `get_candles` in Claude Code, and model training in the notebooks.
- **Usage:** keep the download and MCP server paragraphs. Remove "Everything past a trained model … is Part B and not implemented yet". Point to the PRD for what's coming.
- **Replace "Planned Trade Execution Flow"** with **"How a trading cycle works"**: a short paragraph, plus a Mermaid sequence diagram of the PRD's cycle (`/loop` → agent → `get_candles` → `get_forecast` → `analyze_forecast` → `get_portfolio` / `get_risk_status` → hold or `propose_order` → `place_order` → `record_decision`). The tools that don't exist yet are labelled "planned". The paragraph says the server enforces the risk limits, not the agent, and links to PRD §5–§6.
- **Project Structure tree:**
  - add `.mcp.json`, `Taskfile.yml`, `.python-version` and `src/fartt/server`;
  - change `docs/product` to "The PRD for the MCP trading agent";
  - remove `core` and its children;
  - describe `model` as "Regression pipeline epic B builds on";
  - add `exchange` children (`exchange.py`, `ccxt_exchange.py`, `candle.py`) and `server/server.py`.
- **References:** rename "Part A — Signal generation" to "Model selection (epic B)" and "Part B — Trade execution" to "Risk controls (epic D)". Reword their intros so they point to the PRD and the epics instead of the removed state machine and the deleted PRDs, and keep every citation.
- **Acknowledgements:** keep Mermaid, since the new section uses it.

- [ ] **Step 4: Check that the docs match the code**

Run:

```bash
grep -ohE '`(src/fartt|tests|docs|notebooks)/[^` ]+`|`fartt/[^` ]+`|`\.mcp\.json`|`Taskfile\.yml`|`\.python-version`' CLAUDE.md README.md \
  | tr -d '`' | sed 's#^fartt/#src/fartt/#; s#::.*##' | sort -u \
  | while read -r path; do [ -e "$path" ] || echo "missing: $path"; done; echo done
```

Expected: only `done`. Every file and folder the docs name exists.

Run: `grep -nE "Part A|Part B|core/|broker|dashboard|part-a-|part-b-" CLAUDE.md README.md`
Expected: no matches.

Run: `task check 2>&1 | grep -E "passed|errors,|All checks"`
Expected: `All checks passed!`, `0 errors`, `136 passed, 1 deselected`.

- [ ] **Step 5: Commit and push**

```bash
git add CLAUDE.md README.md
git commit -m "docs: describe the MCP architecture instead of the Part A/B split"
git push
```
