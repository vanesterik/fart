# Part A: ML Signal Generation Refactor — PRD

**Date:** 2026-07-31
**Status:** Draft
**Source:** Adapted from [Problem Framing Canvas](./part-a-signal-generation-refactor.md)

---

## 1. Executive Summary

We're building a regression-based signal generation system for BTC trading — replacing six classic up/down/hold classifiers with a sequence-aware architecture (N-BEATS or time-series transformer) — to solve the problem of signals that discard trade magnitude and ignore trade cost, which caps their reliability. This will produce cost-aware, trustworthy signals that the planned execution system (Part B) can act on with confidence.

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

We're building a **regression-based signal generation pipeline** that replaces the six classifiers with sequence-aware deep learning models. Both **N-BEATS** and a **time-series transformer** will be implemented and evaluated in parallel against the same backtest harness; the better performer (or an ensemble of both) becomes the production model.

### How it works
1. Existing feature engineering (`fart/features/calculate_technical_indicators.py` — Bollinger Bands, EMA, MACD, RSI) continues to feed the model as input, reusing the existing Polars-based pipeline.
2. The existing time-series-respecting split (`fart/model/train_test_split.py`, non-shuffled) is reused.
3. `fart/model/train_model.py` and `predict_model.py` (currently empty stubs) get filled in: training logic for both architectures, and a shared prediction interface that outputs **expected trade magnitude + confidence** rather than a discrete up/down/hold label.
4. A walk-forward backtest (respecting realistic trade costs/slippage) evaluates both models on the same historical data, comparing against the classifier baseline.

### Key features
- Dual-architecture evaluation harness (N-BEATS vs. transformer) with shared metrics
- Regression output format: magnitude + confidence, not a class label — this is the new "contract" Part B will consume
- Backtest-first validation before any live deployment consideration

### Candidate architectures
- N-BEATS ([arXiv:1905.10437](https://arxiv.org/abs/1905.10437))
- Time-Series Transformer ([arXiv:2205.01138](https://arxiv.org/pdf/2205.01138))

---

## 6. Success Metrics

### Primary Metric
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
We believe that replacing classifier-based signal generation with a sequence-aware regression model (N-BEATS or transformer) will produce a backtested net return, after costs, that beats both the classifier baseline and buy-and-hold — because the current approach discards trade magnitude and temporal structure that a regression model can capture. We'll measure this via walk-forward backtest results by end of Q3 2026.

### User Stories

**Story 1: Reuse feature pipeline as model input**
As the operator, I want the existing technical-indicator features (Bollinger, EMA, MACD, RSI) available as sequence input to a regression model, so I don't have to rebuild feature engineering.

**Acceptance Criteria:**
- [ ] `train_model.py` consumes the existing Polars feature DataFrame without modification to `calculate_technical_indicators.py`

**Story 2: Train an N-BEATS model**
As the operator, I want an N-BEATS model trained on historical BTC candles to predict trade magnitude, so I have one working regression baseline.

**Acceptance Criteria:**
- [ ] Model trains on the existing time-series split
- [ ] Outputs a magnitude + confidence prediction per candle

**Story 3: Train a time-series transformer model**
As the operator, I want a transformer model trained the same way, so I can compare against N-BEATS.

**Acceptance Criteria:**
- [ ] Same output contract as the N-BEATS model
- [ ] Trained on the identical split for fair comparison

**Story 4: Walk-forward backtest harness**
As the operator, I want a backtest that applies realistic trade costs/slippage to both models' predictions, so I can compare them on the primary metric.

**Acceptance Criteria:**
- [ ] Harness reuses/extends `TradeStrategy` in `trade_strategy.py`
- [ ] Reports net return, MAE/RMSE, Sharpe, max drawdown for each model plus the classifier baseline and buy-and-hold

**Story 5: Select and wire up the production model**
As the operator, I want the winning model wired into `predict_model.py`, so it's ready to feed Part B once validated.

**Acceptance Criteria:**
- [ ] `predict_model.py` exposes a stable output contract (magnitude + confidence) regardless of which architecture won

---

## 8. Out of Scope

**Not included in this initiative:**
- Building Part B (trade execution, position sizing, kill switch) — deliberately sequenced after Part A
- Live/real-time deployment of the winning model
- Broker connectivity (`fart/core/broker.py` remains a stub)
- Dashboard/visualization work (`fart/core/dashboard.py`) — unrelated to signal generation
- Additional architectures beyond N-BEATS and transformer (e.g. regime-detection ensembles) — future consideration if both underperform

---

## 9. Dependencies & Risks

### Dependencies
- Historical BTC candle data via existing `fart download` / Bitvavo downloader
- Existing feature engineering (`talib`-based indicators) and `train_test_split.py`
- New ML dependencies not yet in `pyproject.toml` (e.g. N-BEATS/transformer libraries) will need to be added

### Risks & Mitigations
- **Risk:** Overfitting on noisy/non-stationary crypto data
  - **Mitigation:** Walk-forward validation, not a single train/test split
- **Risk:** Evaluating two architectures solo in ~2 months is ambitious
  - **Mitigation:** Keep both implementations minimal-viable before optimizing either
- **Risk:** Backtest-reality gap — synthetic P&L already showed optimism bias once
  - **Mitigation:** Insist on realistic cost/slippage assumptions before declaring success

---

## 10. Open Questions

- 🔵 Concrete numeric target for "beat baseline/buy-and-hold" (deferred until first backtest)
- 🔵 If both architectures perform similarly, ensemble them or pick one for simplicity?
- 🔵 How is "confidence" calibrated/validated, not just magnitude?

---

## Self-Assessment

- **Strongest section:** Problem Statement and Strategic Context — directly grounded in the existing Problem Framing Canvas, well-evidenced.
- **Weakest section:** Success Metrics — primary metric target is an open question pending real backtest data; this is the biggest gap to close before treating the PRD as final.
- **Top assumptions to validate:**
  1. 🔶 That evaluating two architectures in parallel is feasible within the 2-month timeline for a solo operator
  2. 🔶 That existing feature engineering (technical indicators) is sufficient input for a sequence model without further feature work
- **Recommended next step:** Run a first-pass walk-forward backtest on the classifier baseline to establish the "current" value for the primary metric, then set a concrete numeric target before starting Story 2/3 implementation.
