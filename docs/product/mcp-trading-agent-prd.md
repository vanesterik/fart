# MCP Trading Agent — PRD

**Date:** 2026-10-07
**Status:** Draft, awaiting approval
**Supersedes:** the Part A (signal generation) and Part B (trade execution) PRDs and framing canvases, deleted with this PRD. The last version is [`docs/product` at 82ee8cf](https://github.com/vanesterik/fart/tree/82ee8cfc6567f2fbb16e6687c0041e4f453db7c3/docs/product).

---

## 1. Executive Summary

We're replacing the plan to build a full trading platform with an MCP server that exposes market data (through ccxt, so Bitvavo isn't a lock-in), a BTC forecast, deterministic forecast analysis, portfolio and performance state, server-enforced risk limits, orders and a decision journal as tools, operated by an agent with Claude Code as the client. The project's effort has been going into platform plumbing (dashboard, broker loop, scheduler, CLI) that needs constant upkeep, and that is why the forecast model hasn't been proven yet, so nothing can safely act on it. Handing orchestration and the operator interface to Claude Code (a `/loop` running the server's trading-cycle prompt) removes the plumbing and moves the effort to the model, the tools and the risk controls.

The forecast model is chosen by screening the own architectures (MLP, CNN, GRU, N-BEATS, time-series transformer) and zero-shot foundation models against naive baselines, then backtesting the best after costs on a 30m, 1h or 4h interval. The system then moves through four stages: paper with approval, paper automatic, live with approval and live automatic, each one earned through a risk-adjusted gate against buy-and-hold BTC. The end state is a small live experiment run with minimal upkeep. If no model earns the first gate, that is still the verdict the current plan can't deliver.

---

## 2. Problem Statement

### Who has this problem?
You, as the solo developer and operator of FART.

### What is the problem?
The project is trying to build two hard things at once: a forecast model that has an edge, and a complete trading platform around it. The platform work (dashboard, broker loop, scheduler, CLI, exchange plumbing) needs constant upkeep. It has taken effort away from proving the model, so after several refactors there is still no trustworthy signal, and no working path from a forecast to an order.

### Why is it painful?
- **The effort produces no answer.** Months of work haven't reached the point where the system shows whether there is an edge, so the project could stall without ever finding out.
- **Forcing the current path through is risky.** Finishing the self-built platform means running real money through code written and maintained by one person, with no tested risk controls.

### Evidence

**Model side: no forecast has been proven yet.**
- The six classic classifiers (AdaBoost, Gradient Boosting, Logistic Regression, k-NN, Neural Net, Random Forest) all performed suboptimally on up/down/hold, which led to the regression refactor (Part A).
- An earlier N-BEATS prototype with beta-NLL loss was deleted: a 130-run reproducibility check found its predicted confidence only weakly informative (r ≈ −0.09) under every configuration tested.
- The MLP and CNN are built and run from notebooks, but the screening hasn't been run to comparable results. GRU, N-BEATS (rebuilt fresh) and a time-series transformer are still to be built.

**Platform side: the self-built platform hasn't reached a working path.**
- `core/broker.py` and `model/predict_model.py` are empty stubs. Nothing downstream of a trained model exists.
- `core/exchange.py` (typed Bitvavo wrapper) and `core/dashboard.py` (live terminal dashboard) were built but never connected to an entrypoint.
- `downloader.py` calls the Bitvavo client directly instead of going through `exchange.py`, so there are two separate Bitvavo integrations, and both are tied to Bitvavo.
- The Part B PRD's risk controls (position sizing, stop-loss, kill switch, failure recovery) have not been built.

## 3. Target Users & Personas

### Primary user: the solo operator (you)
- **Role:** Developer, researcher and only trader. Runs the MCP server, reviews the agent's proposals and approves orders.
- **Goals:** Find out whether the forecast model has a real edge after costs, with as little platform upkeep as possible.
- **Constraints:** One person, limited time; needs risk controls that hold even when not watching.
- **Interface:** Claude Code. Conversation, permission prompts and tool output take the place of a dashboard.

### Secondary user: the agent (Claude Code)
- **Role:** Calls every tool. It fetches data, requests and analyses forecasts, checks the portfolio and proposes orders (and places them in a later phase).
- **Needs:**
  - Tool names and descriptions that make the right call obvious and the wrong call unlikely.
  - Compact, structured outputs (key numbers and states, not raw tables) that keep the context small.
  - Errors that say what went wrong and what to do next (for example "order rejected: exceeds max position of X; current position Y").
  - Clear limits: the agent can see what the risk controls allow but can't change them.
- **Why it's named as a user:** Poorly designed tools are the most likely way an agent-driven system makes a bad call.

## 4. Strategic Context

### Why now?
- **A natural break point.** The MLP and CNN builders are done, and GRU, N-BEATS and the time-series transformer haven't been started. Changing the frame now costs almost nothing in model work, and the remaining models get built against the new setup.
- **The tools have matured.** Claude Code and MCP now make an agent-run operator interface realistic, which they weren't when Part B was planned.
- **Stop sunk cost early.** Part B is only scaffolding. Every week spent on it adds code that this approach would replace.

### End state
A **small live experiment**: if the paper-trading edge holds, run with real money in an amount that could be lost completely without it mattering, to confirm that live results match paper trading. Managing meaningful capital is not a goal of this PRD.

## 5. Solution Overview

A Python **MCP server** owns everything that touches data, the model or money: market data through ccxt, forecasts from the selected model, deterministic forecast analysis, portfolio and performance state, server-enforced risk limits, order placement (simulated in paper mode) and a decision journal. **Claude Code is the agent and the operator interface.** A `/loop` in an interactive session runs the server's trading-cycle prompt once per candle interval. The agent fetches the latest candles, gets and analyses a forecast, checks the portfolio and risk status, decides, and records its decision with reasoning. The operator watches, approves orders through permission prompts, and can stop everything with the kill switch.

**How a cycle works:**
1. `/loop` fires the trading-cycle prompt on the candle interval.
2. The agent calls `get_candles`, then `get_forecast` and `analyze_forecast`.
3. The agent calls `get_portfolio` and `get_risk_status`.
4. The agent decides: hold, or `propose_order` → `place_order` (with an exchange-side stop-loss), depending on the autonomy stage.
5. The agent calls `record_decision`, whatever the outcome.

**Decided: autonomy stages.** Each stage is earned by the one before it (the gates are defined in Section 6):

| Stage | Mode | Who places orders |
|---|---|---|
| 1. Paper, approved | Paper | The agent proposes; the operator approves each `place_order` through Claude Code's permission prompt. |
| 2. Paper, automatic | Paper | `place_order` is allowed without a prompt. This tests unattended behaviour (missed cycles, repeated orders, recovery after errors) at no cost. |
| 3. Live, approved | Live, small amount | Same as stage 1, with real money. |
| 4. Live, automatic | Live, small amount | `place_order` is allowed without a prompt; the server limits and the kill switch remain. This is the end state of this PRD. |

### User journey

**One-time setup**
1. Clone the repository and run `uv sync`. Put the exchange API keys in `.env` (read-only keys are enough for the paper stages).
2. Set the server config: market (`BTC-EUR`), interval, mode (`paper`), and the limits (max position, max order size, daily loss limit, starting paper balance).
3. Register the MCP server in the project's `.mcp.json`, so it starts with every Claude Code session in this directory.
4. Set Claude Code permissions: read tools always allowed; `place_order` and `kill_switch` set to ask.

**Choose the model (offline, before trading)**
5. Download history with the downloader, now going through ccxt.
6. Run the screening: own architectures, foundation models and naive baselines on the same splits. Only models that beat the baselines qualify.
7. Run the walk-forward backtest after costs on the best one or two qualifying models, which also settles the interval.
8. Save the chosen model as the active checkpoint, which `get_model_info` then reports.

**Daily operation (stage 1: paper, approved)**
9. Open Claude Code in the project and start `/loop` on the interval with the server's trading-cycle prompt.
10. Each cycle the agent fetches candles, forecasts, analyses, checks the portfolio and risk, then decides. Most cycles end in a recorded "hold", and the operator doesn't need to do anything.
11. To trade, the agent calls `propose_order` and then `place_order`. Claude Code shows a permission prompt with the order and the reasoning. The operator approves or declines; on approval the server simulates the fill and the stop-loss.
12. Any time, the operator can ask in the session about performance against buy-and-hold (`get_performance`), the reasoning behind any decision (the journal), or the limits (`get_risk_status`).

**When something goes wrong**
13. **The agent misbehaves:** the operator calls the kill switch, or the server trips it when the daily loss limit is hit. Open orders are cancelled and new orders blocked until the operator resets it by hand, outside the agent.
14. **The session dies** (laptop sleeps, terminal closes, Claude Code crashes): the loop and the server stop together. Stop-losses resting on the exchange keep protecting live positions. The operator restarts the session and the loop, and the agent picks up the state from `get_portfolio`.

**Moving up the stages**
15. At the end of the paper period the operator checks the gate criteria (Section 6) with `get_performance` and the journal. If they pass, `place_order` is set to allow: stage 2.
16. Stage 2 runs unattended. The operator checks in occasionally, reads the journal, and watches for missed cycles or repeated orders.
17. If the gates pass: create trading API keys, fund a small amount, switch the config to `live`, and set `place_order` back to ask (stage 3). Once live results match paper within tolerance, set it to allow (stage 4).

**Ongoing upkeep**
18. Retraining is offline and run by the operator: rerun the screening and backtest, and replace the active checkpoint only if the new model beats the current one.
19. Upkeep is otherwise limited to dependency updates. ccxt absorbs exchange API changes, and Claude Code is the interface.

**Decided requirement:** All exchange access (market data and orders) goes through one exchange abstraction. Bitvavo is the first implementation, but nothing outside the abstraction depends on Bitvavo, so other exchanges can be added later without changing the tools.

**Decided: adopt [ccxt](https://github.com/ccxt/ccxt).** It's an established library with one unified API for Bitvavo and over 100 other exchanges. A thin internal interface on top keeps ccxt itself replaceable. `core/exchange.py` and the direct `python_bitvavo_api` calls in `downloader.py` are replaced, not extended.

**Decided: Claude Code sets the pace.** A `/loop` in an interactive Claude Code session runs the agent's cycle on the candle interval, and the cycle starts by calling a tool for the latest candle data. The MCP server is passive: it answers tool calls and starts nothing itself. Because the cycle runs in an interactive session, you are there to approve orders. The tools must not depend on how the cycle is triggered, so a push-based trigger can be added later without changing them.

Not chosen for now: pushing events from the server through Claude Code [channels](https://code.claude.com/docs/en/channels-reference). What's known about channels (checked 2026-10-07), for a later revisit:
- An MCP server declares the experimental `claude/channel` capability and sends `notifications/claude/channel` events. Events queue and are delivered in order, grouped together if the agent is busy.
- **Research preview.** Custom channels need `--dangerously-load-development-channels`, and they work only in interactive sessions, not with `claude -p`.
- Claude Code starts the server as a stdio subprocess, so **the server runs only while the session runs.**
- The documentation and examples use the TypeScript MCP SDK. Whether the Python SDK can declare the capability and send the notification hasn't been checked.
- Permission relay can forward tool-approval prompts, such as an order approval, to another device through a two-way channel.

**Decided: model set.** The forecast tool is backed by whichever model screens best among:
- **Own architectures:** MLP, CNN (both built), GRU, N-BEATS (rebuilt fresh) and a time-series transformer, using the existing `prepare_datasets` → builder → `train_model` → `evaluate_model` pipeline.
- **Pretrained foundation models,** run zero-shot (candidates: Chronos, TimesFM).
- **Naive baselines:** "no change" and "repeat last return".

Every candidate is screened on the same chronological splits and metrics (directional accuracy, RMSE, MAE). A model qualifies only if it beats both naive baselines. The best one or two qualifying models go on to a walk-forward backtest after costs.

**Decided: tool surface.**

| Group | Tool | What it does |
|---|---|---|
| Data | `get_candles` | Latest closed OHLCV candles for a market and interval. Also refreshes the cache. |
| Forecast | `get_forecast` | The selected model's forecast for the next candle. |
| | `analyze_forecast` | Deterministic: expected return after fees and slippage, whether it clears the trade threshold, and the model's recent hit rate. |
| | `get_model_info` | Active model, training date, screening and backtest metrics, and whether the data behind the latest forecast is current. |
| Portfolio | `get_portfolio` | Balances, open position, open orders, and P&L today and in total. |
| | `get_performance` | Paper (later live) P&L against buy-and-hold since the start, after costs. |
| Risk | `get_risk_status` | Current limits, how much of each is used, and whether the kill switch is active. |
| Orders | `propose_order` | Records a proposed order with the agent's reasoning and checks it against the limits. Nothing is placed. |
| | `place_order` | Places an order, always checked against the limits on the server. In paper mode, the server simulates the fill. |
| | `cancel_order` | Cancels an open order. |
| | `kill_switch` | Cancels everything and blocks new orders until the operator resets it by hand. |
| Journal | `record_decision` | Logs the cycle's decision, including "hold", with the forecast, the analysis and the agent's reasoning. |

The server also publishes the **trading cycle as an MCP prompt** (fetch → forecast → analyse → check portfolio and risk → decide → record), and `/loop` runs it. The agent's procedure is versioned in the repository next to the tools.

**Server-side rules:**
- The limits (paper or live mode, max position, max order size, daily loss limit) live in server configuration. The agent can read them through `get_risk_status` but can't change them.
- Stop-losses are placed as orders on the exchange, together with the entry order, so they hold when no session is running.
- `analyze_forecast` is deterministic; the agent applies judgement on top of its numbers.

**Deliberately excluded:** a retraining tool (training stays an offline step the operator runs, so the agent never changes the model it's judged on) and news or sentiment tools (a second signal before the first one is proven).

## 6. Success Metrics

### Primary metric
**Agent P&L after costs against buy-and-hold BTC** over the same period, starting with the paper period.

### Trading interval
**Decided:** the walk-forward backtest of the best qualifying model chooses between **1h and 4h**, with **30m** included as a check. Intervals below 30m are excluded. The 1d interval is excluded from the first iteration because a paper period of a few months gives too few decisions to separate skill from luck.

Evidence: BTC-EUR, 1m candles resampled per interval, 2025-08-04 to 2026-08-04:

| Interval | Median candle move | Candles moving > 0.5% |
|---|---|---|
| 1m | 0.025% | 0.1% |
| 5m | 0.056% | 1.0% |
| 15m | 0.100% | 4.4% |
| 30m | 0.141% | 8.8% |
| 1h | 0.201% | 17.0% |
| 4h | 0.413% | 42.6% |
| 1d | 1.231% | 76.4% |

Why sub-30m is excluded:
- **Fees outweigh the moves.** A round trip costs roughly 0.4–0.5% in taker fees at the entry tier (🔶 **Assumption:** to be confirmed against the actual Bitvavo fee tier), which fewer than 5% of 15m candles reach.
- **Agent speed and cost.** A cycle of tool calls plus reasoning takes tens of seconds and is billed per token, and an approval stage would need constant attention.
- **A different architecture.** High-frequency trading needs a deterministic strategy without an agent in the loop.

### Gate: stage 1 (paper, approved) → stage 2 (paper, automatic)
The agent is long-only, so in a falling market holding cash beats buy-and-hold with no skill involved. The gate is therefore risk-adjusted. Over at least **12 weeks** and at least **30 closed trades**, all of these must hold:
- Return after costs is above buy-and-hold over the same period.
- Sharpe ratio is above buy-and-hold's.
- Maximum drawdown is no worse than buy-and-hold's.
- Results are within tolerance of what the walk-forward backtest predicted for the same period, so the edge matches the validated one.

🔶 **Assumption:** the backtest-match tolerance is still to be set (for example, return and hit rate within the backtest's own variation across walk-forward folds).

### Gate: stage 2 (paper, automatic) → stage 3 (live, approved)
At least **4 weeks** unattended, with:
- The stage 1 bundle still holding over the cumulative paper period.
- At least **98%** of scheduled cycles completed.
- Zero limit breaches reaching the exchange, zero duplicate orders, zero unrecorded decisions.

### Gate: stage 3 (live, approved) → stage 4 (live, automatic)
At least **4 weeks** and at least **10 closed live trades**, with:
- Live fills within the expected slippage assumed by the paper model.
- Live P&L tracking paper behaviour within the same tolerance as the stage 1 backtest match.

### Guardrails
Monitored at every stage. A breach moves the system back one stage.
- The daily loss limit is never exceeded.
- No order is placed without a recorded decision.
- Agent token cost per week stays under an operator-set budget.
- The kill switch has been tested by hand in each stage before moving to the next.

## 7. User Stories & Requirements

### Epic hypothesis
We believe that replacing the self-built platform with an MCP server operated by Claude Code will get a forecast model to a clear paper-trading verdict (pass or fail against the stage 1 gate) with minimal platform upkeep, because the effort moves from plumbing to the model, the tools and the risk controls. We'll know this is true when the stage 1 gate has been evaluated on at least 12 weeks of paper trading. A "fail" verdict also counts, because the current plan produces no verdict at all.

### Epics
Built as a **walking skeleton first**: the whole agent loop runs end to end on a naive-baseline model before the real model is selected. Tool design and loop behaviour get tested early at no cost, and the model from epic B then replaces the baseline.

| Order | Epic | Contents |
|---|---|---|
| 1 | **A. Server foundation** (#44) | MCP server skeleton and registration, ccxt exchange layer, `get_candles`, downloader moved to ccxt. Superseded code removed: `core/exchange.py`, `core/dashboard.py`, `core/broker.py`, direct `python_bitvavo_api` calls. |
| 2 | **C. Forecast tools** (#45) | `get_forecast`, `analyze_forecast`, `get_model_info`, backed by a naive baseline at first. |
| 3 | **D. Paper trading and risk (stage 1)** (#46) | Risk config, `get_portfolio`, `get_risk_status`, `propose_order`, `place_order`, `cancel_order` (simulated fills and stop-losses), `kill_switch`, `record_decision`, `get_performance`, trading-cycle prompt. |
| 4 | **B. Model selection** (#47) | GRU, N-BEATS and transformer builders; foundation models; naive baselines; screening; walk-forward backtest after costs; interval choice (30m, 1h or 4h); active checkpoint replaces the baseline. Starts the 12-week stage 1 paper period. |
| 5 | **E. Unattended operation (stage 2)** (#48) | Session lifetime, recovery after restart, missed-cycle monitoring. |
| 6 | **F. Live trading (stages 3–4)** (#49) | Live orders through ccxt, exchange-side stop-losses, live keys and config. |

### Stories
Written **just in time**: full stories with acceptance criteria become GitHub issues on the project board one epic at a time, just before that epic starts. The titles below are the current breakdown and will be refined as earlier epics teach us more.

**A. Server foundation**
- A1 (#50). Exchange layer: fetch candles through ccxt behind an internal interface
- A2 (#51). Downloader uses the exchange layer (CSV cache and resume kept)
- A3 (#52). MCP server skeleton with `get_candles`, registered in `.mcp.json`
- A4 (#53). Remove superseded code and dependencies (`core/exchange.py`, `core/dashboard.py`, `core/broker.py`, `python_bitvavo_api`)

**C. Forecast tools**
- C1. Forecaster interface with a naive-baseline implementation
- C2. `get_forecast`
- C3. `analyze_forecast`: expected return after fees and slippage, threshold, recent hit rate
- C4. `get_model_info`

**D. Paper trading and risk**
- D1. Risk config and `get_risk_status`
- D2. Paper portfolio state and `get_portfolio`
- D3. `propose_order` and `place_order` with simulated fills and enforced limits
- D4. Simulated stop-losses and `cancel_order`
- D5. `kill_switch` with a manual reset outside the agent
- D6. `record_decision` journal
- D7. `get_performance` against buy-and-hold
- D8. Trading-cycle prompt and permission setup: first stage 1 run on the baseline

**B. Model selection**
- B1. Naive baselines in the screening pipeline
- B2. GRU builder
- B3. N-BEATS builder (rebuilt fresh)
- B4. Time-series transformer builder
- B5. Zero-shot foundation model evaluation (Chronos, TimesFM)
- B6. Screening comparison across all candidates and intervals (30m, 1h, 4h)
- B7. Walk-forward backtest after costs and interval choice
- B8. Promote the selected model behind the forecaster interface

**E. Unattended operation**
- E1. Hosting for a long-running session and the `/loop` lifetime
- E2. Recovery after restart, including stop-losses missed while the server was down
- E3. Missed-cycle monitoring

**F. Live trading**
- F1. Live order placement through ccxt
- F2. Exchange-side stop-losses placed with the entry order
- F3. Live keys, config, and safeguards for switching modes

Notes on the breakdown:
- C1's forecaster interface has two concrete implementations planned (the naive baseline and the model selected in B), so it isn't an early abstraction.
- The 12-week stage 1 paper period starts at **B8**, when the selected model is promoted. The D8 run on the baseline tests the system and doesn't count toward the gate.

## 8. Out of Scope

**Not in this PRD:**
- **Markets other than BTC-EUR.** The exchange layer allows others, but only one market is traded.
- **Exchanges other than Bitvavo.** ccxt keeps them possible, but none are built or tested.
- **Short selling, leverage and derivatives.** The agent is long-only, spot only.
- **Intervals below 30m and high-frequency strategies** (see Section 6).
- **Meaningful capital.** The end state is a small live experiment (see Section 4).
- **Agent-driven retraining and news or sentiment signals** (see Section 5).
- **A custom dashboard or UI.** Claude Code is the interface.
- **Headless runs** (`claude -p` from cron). They leave no one present to approve orders.

**Future consideration:**
- A push-based trigger through Claude Code channels, once out of research preview.
- Ensembles of several models.
- Approving orders from a phone through channel permission relay.

## 9. Dependencies & Risks

### Dependencies
- **ccxt:** Bitvavo support for candles, balances and orders, including stop-loss order types (🔶 **Assumption:** to be verified in A1 and F2).
- **MCP Python SDK:** the server, its tools and the trading-cycle prompt.
- **Claude Code:** `/loop`, permission rules, MCP server registration.
- **Foundation-model packages** (Chronos, TimesFM): heavy dependencies, possibly installed as a screening-only extra.
- **Bitvavo API:** data, orders, rate limits.

### Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | **No model beats the naive baselines.** | That's an acceptable verdict under the epic hypothesis. The walking skeleton keeps the work spent before the verdict small. |
| 2 | **Overfitting through many comparisons.** About 9 candidates across 3 intervals means a winner can come from luck. | The final test split stays untouched until selection; walk-forward backtest; the paper period is the real out-of-sample test. |
| 3 | **The server's lifetime is tied to the Claude Code session** (stop-losses in paper mode, session lifetime). | Live stop-losses rest on the exchange; epic E resolves both gaps before stage 2. |
| 4 | **The agent misbehaves:** wrong size, repeated orders, misread tool output. | Limits enforced on the server, the `propose_order` step, approval stages, the decision journal, and idempotent orders (each order carries a client ID, so a repeat is rejected). |
| 5 | **Broad session permissions:** the trading session also has Bash, file editing and other MCP servers. | Run the loop in a dedicated session or settings profile that allows only this server's tools. |
| 6 | **API key exposure.** | Live keys can trade but not withdraw, are restricted by IP where Bitvavo allows it, and are stored only in `.env`. |
| 7 | **Paper fills are too optimistic.** | Conservative slippage model; gate 3→4 measures live fills against it. |
| 8 | **Market regime change or model drift.** | Guardrails drop the system back a stage; retraining policy is an open question (Section 10). |
| 9 | **Token cost grows.** | A weekly budget as a guardrail; 1h or 4h keeps it to 6–24 cycles a day. |

## 10. Open Questions

Gaps found while writing the user journey (Section 5):
- 🔵 **Session lifetime (stage 2 onwards):** where does an interactive Claude Code session run unattended for weeks, and does `/loop` have a maximum lifetime?
- 🔵 **Simulated stop-losses while the server is down:** in paper mode, stop-losses exist only inside the server, so they don't fire while the session is down. Simulate them on restart from the candles that were missed, or accept the gap?
- 🔵 **Kill switch reset:** how the operator resets it outside the agent (a CLI command or a config edit).
- 🔵 **Retraining:** how often, and what triggers it.

Assumptions tagged inline that need settling:
- 🔶 **Backtest-match tolerance** for the stage gates (Section 6).
- 🔶 **Bitvavo fee tier:** the round-trip cost of roughly 0.4–0.5% used to exclude sub-30m intervals (Section 6).
- 🔶 **ccxt coverage** of Bitvavo stop-loss order types (Section 9).
