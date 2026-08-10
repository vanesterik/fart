# N-BEATS Ensemble Confidence — Design

**Date:** 2026-08-10
**Status:** Approved
**Related:** `notebooks/2.0-kve-nbeats-confidence-calibration.ipynb` (diagnoses the problem this addresses), `notebooks/3.0-kve-beta-nll-reproducibility-check.ipynb` (rules out beta-NLL as a fix)

## Problem

`NBeatsNet` (`src/fart/model/nbeats.py`) predicts `(mu, log_sigma)` per window from a single shared trunk: each `NBeatsBlock`'s 4-layer FC stack feeds one `forecast_layer = nn.Linear(hidden_width, 2)`, so mean and log-variance are just two columns of the same final weight matrix, trained jointly via `beta_nll_loss` with no stop-gradient between them. Notebook 2.0 shows the resulting confidence (`1/(1+exp(log_sigma))`) collapsing to a near-constant band (~0.965–0.976 across `n=535` test windows, full 7-year BTC-EUR 1d history) that is uncorrelated with actual error (Pearson r=0.043). Notebook 3.0's 130-run check already ruled out beta-NLL reweighting as a fix, which is consistent with the cause being architectural (no independent capacity for the variance head) rather than a loss-weighting problem.

The hypothesis to test: if uncertainty is estimated a way that doesn't require the network to introspect on its own confidence — via disagreement across an ensemble of independently-initialized point-forecast models — does the resulting confidence signal correlate with actual error better than the current self-reported one?

## Approach

Deep ensemble (Lakshminarayanan et al. 2017): train 5 independently-initialized `NBeatsNet` instances as plain point predictors (MSE on `mu` only, `log_sigma` output ignored), and derive confidence from the spread of their predictions on each test window, rather than from a learned log-variance.

This is scoped as a **notebook-only experiment** — no changes to `src/fart`. It answers the research question (does ensemble disagreement correlate with error better than self-reported confidence) cheaply, before committing to extracting reusable training/inference code. If the result is positive, extracting an ensemble trainer into `src/fart/model/` is a natural follow-up, not part of this design.

### Data

Identical to notebook 2.0: `prepare_training_data(data_dir=assets_dir, market="BTC-EUR", interval="1d")` — full cached history (~2706 candles, no `months` filter) — then `build_return_windows(close_prices, lookback)` using the same train/test split. Using the same data and split means the two notebooks' test sets match exactly, so their confidence-vs-error plots are directly comparable and any difference is attributable to the estimation approach, not a data change.

### Ensemble training

5 members, same `NBeatsConfig` defaults used to train the model in notebook 2.0 (no hyperparameter variation across members — diversity comes only from random weight init and minibatch shuffling, per the standard deep-ensembles recipe). Each member is a fresh `NBeatsNet(config)`, trained in the notebook with a loop shaped like `train_model.py::train()`'s (`DataLoader`/`TensorDataset`, `Adam`, same `epochs`/`batch_size`), but with `nn.MSELoss()(mu, y_batch)` in place of `beta_nll_loss`, where `mu = model(X_batch)[..., 0]`. `log_sigma` (`[..., 1]`) is computed by the forward pass but never appears in the loss, so those weight-matrix columns receive no gradient and stay at random init — harmless, since they're discarded. This reuses `NBeatsNet` unmodified rather than requiring a single-output variant, at the cost of a wasted (but inert) output column.

Each trained member's predictions on the test windows are collected into a `(5, n_test)` tensor.

### Aggregation and confidence metric

Per test window:
- `mu_ensemble` = mean of the 5 members' predictions
- `sigma_ensemble` = std of the 5 members' predictions
- `error = |y_test - mu_ensemble|` — same definition as notebook 2.0's error
- `confidence = 1 / (1 + sigma_ensemble)` — the same squashing formula the current model already uses for `confidence = 1/(1+sigma)` where `sigma = exp(log_sigma)`, just fed an empirical std instead of a learned one. This keeps scale/units comparable to notebook 2.0 and means the existing `plot_confidence_calibration(confidence, error)` (`src/fart/visualization/confidence_calibration.py`) can be reused unmodified.

### Notebook structure

New file `notebooks/4.0-kve-nbeats-ensemble-confidence.ipynb`, following the existing `N.0-kve-<topic>` numbering. Cells, mirroring notebook 2.0's shape:

1. Imports / `apply_plot_styles()`
2. Load and prepare data (as above)
3. Train 5 ensemble members in a loop, collecting per-member test-window predictions
4. Aggregate into `mu_ensemble`, `sigma_ensemble`, `error`, `confidence`
5. `plot_confidence_calibration(confidence, error)`
6. Markdown note comparing this run's Pearson r and confidence spread against notebook 2.0's (r=0.043, band 0.965–0.976)

## Out of Scope

- Any change to `src/fart` (`nbeats.py`, `train_model.py`, etc.) — purely additive notebook.
- Varying architecture/hyperparameters across ensemble members.
- Persisting the 5 trained members as artifacts (`nbeats_persistence.py` is not used here — this is a one-off research notebook, not a trained-and-saved model).
- Extracting an ensemble-training helper into `src/fart/model/` — a follow-up only if this experiment's result supports the approach.
- Testing/CI — no unit tests, consistent with `tests/` excluding notebook logic elsewhere in this repo.

## Open Questions

None — data, ensemble size, member diversity, confidence metric, and code placement were all confirmed during design.
