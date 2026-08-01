# Part B: Trade Execution System — PRD

**Date:** 2026-08-01
**Status:** Draft
**Source:** Adapted from [Problem Framing Canvas](../canvas/part-b-trade-execution-system.md)

---

## 1. Executive Summary

We're building a single-exchange trade execution system — designed as a risk-management problem first (position sizing, stop-loss, kill switch) and a connector-building problem second — for the solo operator of FART, to solve the problem that no system exists to safely turn validated trading signals into real trades. This will let signals become real-world action without the execution layer itself becoming a new source of loss.

---

## 2. Problem Statement

### Who has this problem?
You, as the solo operator with capital at risk; indirectly Part A, since execution is what turns a signal into a real-world consequence.

### What is the problem?
No system exists to act on trading signals in real time. Manual trade placement is too slow, and there's no automated execution layer — scoped to a single exchange.

### Why is it painful?
A poorly-built execution layer can turn a good signal into a loss (slippage, bad fills, missed timing), or turn a failure into a much bigger loss (no stop-loss, no kill switch, silent errors during exchange downtime, partial fills, or rate limits).

### Evidence
🔶 **Assumption** — unlike Part A, there's no backtest or interview data here yet; the case rests on reasoning from the canvas (retail algo traders commonly need circuit breakers, position limits, idempotent order handling) rather than FART-specific evidence, since nothing has been built or run live yet.

---

## 3. Target Users & Personas

### Primary persona: You, the operator
- **Role:** Solo trader running FART end-to-end, with personal capital at risk
- **Tech savviness:** High (you built the system) but no ops team backing you up
- **Goals:** Let validated signals convert into trades without babysitting every tick; trust that safeguards will catch failures you didn't anticipate
- **Pain points:** Manual execution is too slow to act on signals in time; automating without safeguards risks losing money faster and more silently than trading manually
- **Behavior today:** Not currently executing trades automatically at all — this system doesn't exist yet

### Secondary persona
🔶 **Assumption** — none identified distinct from the primary. Unlike Part A (which had Part B as a downstream software consumer), Part B's only "consumer" is you, acting on Part A's signals. If a secondary matters, it would be "future-you during an incident" — same person, different mode (monitoring/recovery instead of steady-state).

---

## 4. Strategic Context

### Goal
Convert validated Part A signals into safe, automated real-world trades on a single exchange, without the execution layer itself becoming a new source of loss.

### Competitive landscape
Retail algo traders on retail exchange APIs commonly rely on circuit breakers/kill switches, paper trading before going live, position size limits, idempotent order handling, exchange rate-limit backoff, and logging/alerting on every order state change — Part B should meet this same bar, not invent it from scratch.

### Why now
Deliberately *not* now in the sense of "build immediately" — Part B is explicitly sequenced **after** Part A's signals are validated (per the canvas reframe: risk-management problem first, connector-building problem second). Starting Part B before Part A is trustworthy would compound Part A's uncertainty with real capital.

### Target date
🔵 **Open Question** — no fixed date. Part B's start is contingent on Part A completing (targeted end of Q3 2026); a concrete Part B timeline should be set once that gate clears.

---

## 5. Solution Overview

We're building a **single-exchange trade execution system**, sequenced deliberately as risk-management first, connector second, per the canvas reframe.

### How it works
1. **Risk-management layer** (built first): position sizing rules, stop-loss, max-drawdown kill switch — these exist and are testable *before* any order is ever placed against a real exchange.
2. **Failure-mode handling**: explicit handling for exchange downtime, partial fills, API auth expiry, and rate limits — treated as first-class design inputs, not edge cases bolted on after the connector works.
3. **Paper-trading / dry-run mode**: the system runs against live market data and Part A's real signal output, but places no real orders — used to validate risk controls and failure handling before any capital is at risk.
4. **Exchange connector**: order placement wired into `fart/core/broker.py` (currently a stub), built on the existing typed `fart/core/exchange.py` wrapper around the Bitvavo REST/websocket client.
5. Consumes Part A's output contract directly (expected magnitude + confidence per candle).

### Key features
- Kill switch that halts trading on max-drawdown breach or repeated failures
- Idempotent order handling (safe to retry without double-placing orders)
- Dry-run/paper-trading mode as a first-class, not bolted-on, operating mode
- Logging/alerting on every order state change

---

## 6. Success Metrics

### Primary Metric
**Max drawdown stays within a defined bound**, in both paper-trading and (eventually) live trading.
- **Current:** Not applicable — no execution system exists yet
- **Target:** 🔵 **Open Question** — concrete drawdown threshold to be set before paper trading begins (informed by Part A's guardrail metric and personal risk tolerance)

### Secondary Metrics
- Paper-trading net return vs. Part A's backtest expectation (execution fidelity — how closely live paper results track what the backtest promised)
- Kill-switch correctness: fires when it should, never fires on a false positive

### Guardrail Metrics
- Zero silent failures — 100% of exchange downtime/partial fills/API errors/rate-limit events are logged and alerted, none pass unnoticed

---

## 7. User Stories & Requirements

### Epic Hypothesis
We believe that building risk controls (position sizing, stop-loss, max-drawdown kill switch) and a paper-trading validation mode *before* any live connector will let us safely convert Part A's validated signals into real trades without the execution system itself becoming a new source of loss — because unmanaged automation risk (silent failures, unbounded position size, no circuit breaker) is a bigger threat than signal quality alone. We'll measure this via max drawdown staying within defined bounds throughout paper trading.

### User Stories

**Story 1: Position sizing & stop-loss rules**
As the operator, I want every trade sized and stop-loss-bounded by configurable risk rules, so no single trade can exceed my risk tolerance.

**Acceptance Criteria:**
- [ ] Position size is computed from a configurable risk rule, not hardcoded
- [ ] Every simulated/live trade carries a stop-loss bound derived from that rule

**Story 2: Max-drawdown kill switch**
As the operator, I want trading to halt automatically if drawdown breaches a threshold, so a losing streak can't run unchecked.

**Acceptance Criteria:**
- [ ] Kill switch halts new trade placement when max drawdown exceeds the configured threshold
- [ ] Kill-switch state is logged and requires explicit operator action to resume trading

**Story 3: Failure-mode handling**
As the operator, I want exchange downtime, partial fills, API auth expiry, and rate limits handled explicitly, so failures are visible and recoverable, not invisible losses.

**Acceptance Criteria:**
- [ ] Each failure mode (downtime, partial fill, auth expiry, rate limit) has explicit handling logic, not a generic catch-all
- [ ] Every handled failure is logged and alertable, none fail silently

**Story 4: Paper-trading / dry-run mode**
As the operator, I want to run the full system against live market data and real Part A signals without placing real orders, so I can validate risk controls before capital is at risk.

**Acceptance Criteria:**
- [ ] System consumes live market data and live Part A signal output in dry-run mode
- [ ] No real orders are placed against the exchange while in dry-run mode
- [ ] Risk-layer behavior (sizing, stop-loss, kill switch) is fully exercised in dry-run mode

**Story 5: Live exchange connector**
As the operator, I want `fart/core/broker.py` wired to place real, idempotent orders via the existing `fart/core/exchange.py` wrapper, so validated paper-trading behavior can go live.

**Acceptance Criteria:**
- [ ] `broker.py` places real orders through `fart/core/exchange.py`, not a new/duplicate client
- [ ] Order placement is idempotent — retrying a request never double-places an order
- [ ] Live connector only activates after the risk layer, failure handling, and paper-trading validation are in place

---

## 8. Out of Scope

**Not included in this initiative:**
- Multi-exchange support — canvas explicitly scopes to a single exchange
- Part A itself (signal generation) — separate initiative, precedes this
- Portfolio-level / multi-asset risk management — future consideration beyond per-trade sizing
- Dashboard/visualization wiring (`fart/core/dashboard.py`) — not required for execution to function; can integrate later

---

## 9. Dependencies & Risks

### Dependencies
- Part A's signals validated and trustworthy — hard gate before live connector work starts
- Existing `fart/core/exchange.py` typed Bitvavo wrapper (already built)
- `fart/core/broker.py` (currently a stub) — needs to be filled in
- Bitvavo API credentials and its rate limits

### Risks & Mitigations
- **Risk:** Starting before Part A is validated compounds signal uncertainty with real capital
  - **Mitigation:** hard sequencing gate, no live connector work until Part A ships
- **Risk:** Operational complexity (monitoring, error handling, failure recovery) is harder than "place an order via API"
  - **Mitigation:** failure-mode handling and paper trading built before the connector, not after
- **Risk:** The kill switch itself could be wrong — false negative (doesn't fire when it should) or false positive (fires too eagerly)
  - **Mitigation:** dedicated testing of kill-switch logic in paper trading before going live

---

## 10. Open Questions

- 🔵 Concrete numeric drawdown threshold for the kill switch (deferred to Section 6 resolution)
- 🔵 How long/what criteria determine paper trading is "done" and ready to go live? (time-based vs. N clean trades)
- 🔵 Position sizing model — fixed %, Kelly criterion, volatility-based?

---

## Self-Assessment

- **Strongest section:** Solution Overview and User Stories — the risk-first, connector-second sequencing from the canvas translates cleanly into a concrete build order (risk layer → failure handling → paper trading → live connector).
- **Weakest section:** Problem Statement's Evidence — unlike Part A, there's no data or interview backing this yet; it's reasoning from first principles and general retail-algo-trading practice, not FART-specific evidence.
- **Top assumptions to validate:**
  1. 🔶 That there's no meaningful secondary persona distinct from the operator
  2. 🔶 That the existing `fart/core/exchange.py` wrapper is sufficient for order placement without further extension
- **Recommended next step:** Once Part A's target date firms up, set the concrete drawdown threshold and paper-trading "done" criteria (both currently open questions) before starting Story 1 implementation — these gate the whole risk layer's design.
