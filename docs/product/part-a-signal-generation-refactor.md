# Problem Framing Canvas: FART Trading Platform

**Date:** 2026-07-30

---

## Part A: ML Signal Generation Refactor

### Look Inward

**What is the problem? (Symptoms)**
Six classic ML classifiers (AdaBoost, Gradient Boosting, Logistic Regression, k-NN, Neural Net, Random Forest) predicting BTC up/down/hold all perform suboptimally. Want to shift from classification to regression to weigh trade magnitude against trade costs.

**Why haven't we solved it?**
- It's new — limited iteration past the first classifier round
- It's hard — crypto price action is noisy/non-stationary
- Wrong problem framing — classification discards magnitude information that regression would capture

**How are we part of the problem? (Assumptions & biases)**
- Solution-first thinking — tried six algorithms before validating that classification was the right task framing
- Optimizing for ease of implementation over fit — chose off-the-shelf classifiers because they're quick to stand up, not because they suit sequential/temporal price data

**Which of these might be redesigned, reframed, or removed?**
The task framing itself (classification) should be redesigned as regression; the algorithm choice should be redesigned around architectures built for sequential/temporal data.

---

### Look Outward

**Who experiences the problem?**
- **Who:** You, as trader/operator; Part B (the execution system) as a direct downstream consumer of these signals; your capital, directly at risk
- **When/where:** Every live signal, especially during volatile/regime-shift periods
- **Consequences:** False signals → real trading losses; missed signals → opportunity cost; unreliable signals block confident construction of Part B

**Who else has this problem?**
- **Who else:** Other retail/algo crypto traders using classic ML face the same ceiling
- **How they deal with it:** Institutional quants have moved to LSTM/transformer-based time-series models, gradient-boosted regression on engineered features, and regime-detection ensembles

**Who doesn't have it?**
Firms using regression + expected-value frameworks instead of naive up/down/hold classification — the direction this refactor is heading. Candidate architectures identified: MLP, CNN, GRU, N-BEATS ([arXiv:1905.10437](https://arxiv.org/abs/1905.10437)), and Time-Series Transformers ([arXiv:2205.01138](https://arxiv.org/pdf/2205.01138)) — screened in a fast first pass before backtest validation of the best performer(s).

**Who's been left out?**
Rigorous validation — a walk-forward/backtest with realistic trade costs and slippage — hasn't been fully stress-tested yet. Synthetic profit/loss calculations showed a large potential profit (taken with appropriate skepticism, but directionally promising).

**Who benefits?**
- **When problem exists:** Classic ML's simplicity — fast to train, easy to debug, low compute cost
- **When problem doesn't exist:** That simplicity is traded for more compute, more data requirements, and more overfitting risk — but for a signal source worth trusting

---

### Reframe

**Stated another way, the problem is:**
You struggle to generate reliable, cost-aware BTC trading signals because the current approach frames the task as independent up/down/hold classification across six classic ML models, discarding trade magnitude and temporal/sequential structure in the price data. This caps signal accuracy and prevents you from weighing a trade's expected payoff against its costs before acting — and it blocks you from building a trustworthy execution layer (Part B) on top of it. This has been overlooked because the initial approach reached for familiar, easy-to-implement classifiers before validating that classification was the right task framing for genuinely noisy, sequential crypto price data, despite early synthetic backtesting hinting at real profit potential if the signal were captured properly.

**How Might We...**
How might we reframe BTC signal generation as a temporal regression task — screening modern deep learning architectures (MLP, CNN, GRU, N-BEATS, time-series transformers) that capture sequence structure and trade magnitude — as we aim to produce cost-aware, trustworthy real-time signals that a downstream execution system can confidently act on?
