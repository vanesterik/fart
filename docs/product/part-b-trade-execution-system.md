# Problem Framing Canvas: FART Trading Platform

**Date:** 2026-07-30

---

## Part B: Trade Execution System

### Look Inward

**What is the problem? (Symptoms)**
No system exists to automatically act on trading signals in real time. Manual trade placement is too slow, and there's an underlying concern about trusting an automated system to move real money without safeguards. Scope: a single exchange (not multi-exchange).

**Why haven't we solved it?**
- It's new — Part A (signal generation) needed to exist/mature first
- It's hard — real-time execution with reliability/risk controls is genuinely complex
- Sequencing risk — building execution on top of signals that weren't yet trustworthy felt premature

**How are we part of the problem? (Assumptions & biases)**
- Assuming signals will be trustworthy enough to automate on — building execution before Part A is validated risks compounding Part A's uncertainty with real capital
- Underestimating operational risk — assuming "place a trade via API" is the hard part, when monitoring, error handling, and failure recovery (exchange downtime, partial fills, rate limits) may be harder
- Solution-first thinking — jumping to "build execution engine" before defining what risk controls (position sizing, stop-loss, max drawdown) must exist first

**Which of these might be redesigned, reframed, or removed?**
Reframe execution as a risk-management problem first, connector-building problem second. Define risk controls before writing order-placement code.

---

### Look Outward

**Who experiences the problem?**
- **Who:** You directly (capital at risk); indirectly Part A, since execution is what turns a signal into a real-world consequence
- **When/where:** Every time a signal fires and needs to become an order, especially during exchange outages, high volatility, or network/API failures
- **Consequences:** A bad execution layer can turn a good signal into a loss (slippage, bad fills, missed timing), or turn a failure into a bigger loss (no stop-loss, no kill switch, silent errors)

**Who else has this problem?**
- **Who else:** Retail algo traders building on retail exchange APIs; anyone doing solo automated trading without an ops team behind them
- **How they deal with it:** Circuit breakers/kill switches, paper trading before going live, position size limits, idempotent order handling, exchange rate-limit backoff, logging/alerting on every order state change

**Who doesn't have it?**
Institutional desks with dedicated infra/ops teams and formal risk management; also, anyone trading manually has no automation risk (but is slow).

**Who's been left out?**
Failure-mode mapping (partial fills, exchange downtime, API auth expiry, bad signal during a flash crash) and a paper-trading/dry-run plan before live capital is at risk — still to be designed.

**Who benefits?**
- **When problem exists:** Nobody really benefits — but manual trading is at least a known, bounded risk
- **When problem doesn't exist (if solved carelessly):** A poorly-built execution system is worse than no automation — it can lose money faster and more silently than a human would

---

### Reframe

**Stated another way, the problem is:**
You need a way to turn trading signals into safe, reliable real-time trades on a single exchange, but no execution layer exists yet — and building one is riskier than it first appears, because "placing an order via API" is the easy part while monitoring, error handling, and failure recovery (exchange downtime, partial fills, rate limits) is the hard part. Without deliberate risk controls (position sizing, stop-loss, kill switch) designed in from the start, an automated system can turn a good signal into a loss through poor execution, or turn a failure into a much bigger loss through silence. This has been under-planned because execution has been framed as a build-the-connector problem rather than a risk-management problem, and optimistic early backtest numbers from Part A risk biasing toward under-building safeguards.

**How Might We...**
How might we build a single-exchange trade execution layer with risk controls (position sizing, stop-loss, kill switch) and robust failure handling designed in from the start — as we aim to safely convert validated trading signals into real trades without the automation itself becoming a new source of loss?
