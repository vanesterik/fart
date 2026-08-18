# Part A: ML Signal Generation Refactor — PRD

**Date:** 2026-07-31
**Status:** Draft
**Source:** Adapted from [Problem Framing Canvas](./part-a-signal-generation-refactor.md)

---

## 1. Executive Summary

We're building a regression-based signal generation system for BTC trading — replacing six classic up/down/hold classifiers with a sequence-aware architecture — to solve the problem of signals that discard trade magnitude and ignore trade cost, which caps their reliability. Five candidate architectures (MLP, CNN, GRU, N-BEATS, time-series transformer) are screened in a fast first pass, and the 1–2 best performers go through full walk-forward backtest validation before one is selected. This will produce cost-aware, trustworthy signals that the planned execution system (Part B) can act on with confidence.

---

## 2. Problem Statement

### Who has this problem?
You, as the solo trader/operator of FART, plus Part B (the execution system), which depends on these signals as its direct input.

### What is the problem?
Six classic ML classifiers (AdaBoost, Gradient Boosting, Logistic Regression, k-NN, Neural Net, Random Forest) predict BTC up/down/hold, but all perform suboptimally — the classification framing discards trade magnitude and the temporal/sequential structure of price data.

### Why is it painful?
False signals cause real trading losses; missed signals cause opportunity cost; unreliable signals block confident construction of Part B. The task framing itself (classification) is the root issue, not the specific algorithms tried.

### Evidence
Synthetic profit/loss calculations under the current approach showed a large potential profit, taken with appropriate skepticism — directionally promising but not yet stress-tested with a realistic walk-forward backtest (trade costs, slippage).

---

## 3. Target Users & Personas

### Primary "user": You (operator/researcher)
Building, training, and validating these models yourself; need signals you can trust enough to eventually risk capital on.

### Secondary "user": Part B (execution system)
A downstream software consumer, not a person. It needs signals in a specific, stable format (expected magnitude + confidence) it can act on programmatically — this constrains Part A's output contract.

---

## 4. Strategic Context

### Goal
Produce signal quality trustworthy enough to justify building Part B (trade execution) on top of it. Part A is deliberately sequenced before Part B — signals must be validated before risk-management infrastructure is worth building.

### Competitive landscape
Institutional quants have already moved past classifier-based approaches to LSTM/transformer time-series models, gradient-boosted regression on engineered features, and regime-detection ensembles — this refactor brings the project in line with that state of the art.

### Why now
The classifier approach has plateaued after the first full iteration round; continuing to tune six classifiers is unlikely to close the gap, since the ceiling is the task framing itself, not hyperparameters.

### Target date
End of Q3 2026 (~2 months from PRD date) for Part A to be considered done — signals validated, ready to start Part B.

---

## 5. Solution Overview

We're building a **regression-based signal generation pipeline** that replaces the six classifiers with sequence-aware deep learning models, evaluated in two stages:

1. **Screening** — all five candidate architectures are trained and evaluated on the same lightweight train/test split, using directional accuracy and RMSE (the same harness already built for the MLP baseline). This narrows the field to 1–2 finalists without paying full backtest cost for every candidate.
2. **Backtest validation** — only the screening finalist(s) go through a full walk-forward backtest with realistic trade costs/slippage. Whichever wins on the primary metric (backtested net return) becomes the production model.

### How it works
1. Existing feature engineering (`fart/features/calculate_technical_indicators.py` — Bollinger Bands, EMA, MACD, RSI) continues to feed the model as input, reusing the existing Polars-based pipeline.
2. The existing time-series-respecting split (`fart/model/prepare_datasets.py::train_test_split`, non-shuffled) is reused for screening.
3. Each candidate architecture gets its own notebook, adapted from the MLP baseline notebook (`notebooks/2.0-kve-data-analysis.ipynb`) as the template — reusing `prepare_datasets.py`/`evaluate_model.py`/`persist_model.py` where the architecture's input/output shape allows. Screening metrics (RMSE, directional accuracy) come from `evaluate_model.py`, same as the MLP baseline.
4. `fart/model/train_model.py` and `predict_model.py` get filled in for the finalist(s) only: production training logic and a shared prediction interface that outputs **expected trade magnitude** (confidence, where the architecture naturally supports it — see Open Questions) rather than a discrete up/down/hold label.
5. A walk-forward backtest (respecting realistic trade costs/slippage) evaluates the finalist(s) against the classifier baseline and buy-and-hold.

### Key features
- Two-stage evaluation: a fast five-way screen, then full backtest rigor only for the finalists — keeps the expensive part of the work scoped down
- Regression output format: magnitude (+ confidence where supported), not a class label — this is the new "contract" Part B will consume
- Backtest-first validation before any live deployment consideration

### Candidate architectures
- MLP — implemented (`fart/model/train_model.py::build_model`), serves as both the first baseline and the notebook template the other four are adapted from
- CNN
- GRU
- N-BEATS ([arXiv:1905.10437](https://arxiv.org/abs/1905.10437)) — a prior N-BEATS implementation was built and later retired over an unresolved confidence-calibration problem (see project history); it's back in scope here as one candidate among five, not as a committed direction, and is being rebuilt fresh from the notebook template rather than restoring the old module
- Time-Series Transformer ([arXiv:2205.01138](https://arxiv.org/pdf/2205.01138))

---

## 6. Success Metrics

Metrics are unchanged from the original two-architecture plan — expanding to five candidates changes *how many* models get measured, not *what* they're measured on. What's new is that they're applied in two stages (see Solution Overview): screening metrics narrow five candidates to 1–2, then the primary metric makes the final call among those.

### Screening Metrics (Stage 1 — narrows 5 candidates to 1–2 finalists)
- **RMSE** and **directional accuracy** on the held-out test split, via `evaluate_model.py` — same evaluation already used for the MLP baseline.
- 🔵 **Open Question:** exact rule for combining RMSE + directional accuracy into a single narrowing decision isn't defined yet (e.g. best RMSE, best accuracy, or some combination) — needs deciding before the screening round is actually run.

### Primary Metric (Stage 2 — final selection among finalists)
**Backtested net return after realistic trade costs/slippage**, walk-forward validated.
- **Current:** Not established — no rigorous backtest exists yet for the classifier baseline
- **Target:** 🔵 **Open Question** — must beat both the classifier baseline and buy-and-hold after costs; concrete % target to be set once first backtest results exist

### Secondary Metrics
- **Prediction error (MAE/RMSE)** on trade magnitude — model-quality signal during training, independent of trading costs
- **Sharpe ratio** — risk-adjusted return, tracked but not optimized for initially

### Guardrail Metrics
- **Max drawdown** in backtest — shouldn't be worse than the classifier baseline's, since a "better" model that risks larger losses isn't actually trustworthy for Part B

---

## 7. User Stories & Requirements

### Epic Hypothesis
We believe that replacing classifier-based signal generation with a sequence-aware regression model — screened across five candidate architectures (MLP, CNN, GRU, N-BEATS, transformer) and validated via walk-forward backtest — will produce a backtested net return, after costs, that beats both the classifier baseline and buy-and-hold — because the current approach discards trade magnitude and temporal structure that a regression model can capture. We'll measure this via walk-forward backtest results by end of Q3 2026.

### User Stories

**Story 1: Reuse feature pipeline as model input**
As the operator, I want the existing candle-derived features available as input windows to a regression model, so I don't have to rebuild feature engineering per architecture.

**Acceptance Criteria:**
- [x] `prepare_datasets.py` builds sliding input windows from the existing feature pipeline without modifying `calculate_technical_indicators.py`/`calculate_magnitude.py`

**Story 2: Train an MLP baseline**
As the operator, I want a simple feed-forward model trained on historical BTC candles to predict trade magnitude, so I have a working regression baseline and a reusable notebook template for the other four architectures.

**Acceptance Criteria:**
- [x] Model trains on the existing time-series split (`prepare_datasets.py`)
- [x] Outputs a magnitude prediction per candle, screened via `evaluate_model.py` (RMSE + directional accuracy)
- [x] Notebook workflow (`notebooks/2.0-kve-data-analysis.ipynb`) established as the template for adapting to the other four architectures

**Story 3: Train a CNN model**
As the operator, I want a CNN model trained the same way as the MLP baseline, so I can screen a convolutional architecture on the same metrics.

**Acceptance Criteria:**
- [ ] Adapted from the MLP notebook template
- [ ] Screened via `evaluate_model.py` (RMSE + directional accuracy) on the identical split

**Story 4: Train a GRU model**
As the operator, I want a GRU model trained the same way as the MLP baseline, so I can screen a recurrent architecture on the same metrics.

**Acceptance Criteria:**
- [ ] Adapted from the MLP notebook template
- [ ] Screened via `evaluate_model.py` (RMSE + directional accuracy) on the identical split

**Story 5: Train an N-BEATS model**
As the operator, I want an N-BEATS model trained the same way as the MLP baseline, so I can screen it alongside the other candidates on the same metrics.

**Acceptance Criteria:**
- [ ] Built fresh from the notebook template (not restored from the retired `nbeats.py` module)
- [ ] Screened via `evaluate_model.py` (RMSE + directional accuracy) on the identical split

**Story 6: Train a time-series transformer model**
As the operator, I want a transformer model trained the same way as the MLP baseline, so I can screen it alongside the other candidates on the same metrics.

**Acceptance Criteria:**
- [ ] Adapted from the MLP notebook template
- [ ] Screened via `evaluate_model.py` (RMSE + directional accuracy) on the identical split

**Story 7: Screen all five candidates and select finalist(s)**
As the operator, I want all five architectures compared on identical screening metrics, so I can narrow to 1–2 finalists before investing in full backtest rigor.

**Acceptance Criteria:**
- [ ] All 5 models trained on the identical split for fair comparison
- [ ] RMSE + directional accuracy reported for each via `evaluate_model.py`
- [ ] 1–2 finalists selected per the narrowing rule (see Success Metrics § Screening — still an open question)

**Story 8: Walk-forward backtest harness**
As the operator, I want a backtest that applies realistic trade costs/slippage to the finalist(s)' predictions, so I can make the final production-model decision on the primary metric.

**Acceptance Criteria:**
- [ ] Harness reuses/extends `calculate_trade_returns.py`
- [ ] Reports net return, MAE/RMSE, Sharpe, max drawdown for the finalist(s) plus the classifier baseline and buy-and-hold

**Story 9: Select and wire up the production model**
As the operator, I want the winning model wired into `predict_model.py`, so it's ready to feed Part B once validated.

**Acceptance Criteria:**
- [ ] `predict_model.py` exposes a stable magnitude output contract regardless of which architecture won — 🔶 assumes confidence is optional/architecture-dependent rather than a required part of the contract, since not all five candidates naturally produce one (see Open Questions)

---

## 8. Out of Scope

**Not included in this initiative:**
- Building Part B (trade execution, position sizing, kill switch) — deliberately sequenced after Part A
- Live/real-time deployment of the winning model
- Broker connectivity (`fart/core/broker.py` remains a stub)
- Dashboard/visualization work (`fart/core/dashboard.py`) — unrelated to signal generation
- Full walk-forward backtest rigor for all five candidate architectures — only the 1–2 screening finalists get that treatment; the other candidates are evaluated on screening metrics only (see Solution Overview)
- Additional architectures beyond the five screened (MLP, CNN, GRU, N-BEATS, transformer) — e.g. regime-detection ensembles — future consideration only if none of the five perform adequately

---

## 9. Dependencies & Risks

### Dependencies
- Historical BTC candle data via existing `fart download` / Bitvavo downloader
- Existing feature engineering (`talib`-based indicators) and `prepare_datasets.py::train_test_split`
- New ML dependencies not yet in `pyproject.toml` (e.g. N-BEATS/transformer libraries) will need to be added

### Risks & Mitigations
- **Risk:** Overfitting on noisy/non-stationary crypto data
  - **Mitigation:** Walk-forward validation for the finalist(s), not a single train/test split
- **Risk:** Evaluating five architectures solo within the original timeline is ambitious (up from two)
  - **Mitigation:** Two-stage process — a lightweight screen (reusing the existing `evaluate_model.py` harness) across all five, with full walk-forward backtest rigor reserved for the 1–2 finalists — keeps the expensive part of the work scoped down
- **Risk:** Architecture-specific complexity — CNN/GRU/N-BEATS/transformer each have different input-shape and hyperparameter requirements than the MLP template, so adapting the notebook per architecture may take longer than a copy-paste
  - **Mitigation:** Keep each adaptation minimal-viable for screening; don't tune hyperparameters before the narrowing decision
- **Risk:** Backtest-reality gap — synthetic P&L already showed optimism bias once
  - **Mitigation:** Insist on realistic cost/slippage assumptions before declaring success

---

## 10. Open Questions

- 🔵 Concrete numeric target for "beat baseline/buy-and-hold" (deferred until first backtest)
- 🔵 Exact rule for narrowing 5 screened candidates to 1–2 finalists (best RMSE? best directional accuracy? some combination?)
- 🔵 If the screening finalists perform similarly in the backtest, ensemble them or pick one for simplicity?
- 🔵 Does every architecture need to produce a confidence output, or is magnitude-only acceptable for architectures that don't naturally support it (MLP as built, likely CNN/GRU)? The prior N-BEATS attempt's confidence output was never successfully calibrated (r ≈ −0.09 to actual error across every configuration tested) — worth deciding whether confidence is worth pursuing again at all, independent of which architecture wins.

---

## Self-Assessment

- **Strongest section:** Problem Statement and Strategic Context — directly grounded in the existing Problem Framing Canvas, well-evidenced.
- **Weakest section:** Success Metrics — primary metric target is an open question pending real backtest data, and the screening-to-finalist narrowing rule isn't defined yet either; these are the biggest gaps to close before treating the PRD as final.
- **Top assumptions to validate:**
  1. 🔶 That screening five architectures, then backtesting only 1–2 finalists, is feasible within the original timeline for a solo operator
  2. 🔶 That existing feature engineering (technical indicators) is sufficient input for every candidate architecture without further feature work
  3. 🔶 That a magnitude-only output contract (confidence optional/architecture-dependent) is acceptable to Part B, given the prior N-BEATS attempt never got confidence calibration working
- **Recommended next step:** Decide the screening narrowing rule (Open Questions), then run all five architectures through the existing `evaluate_model.py` harness to get real screening numbers before committing further design effort to the backtest harness (Story 8).
