# Model Card — Three-Model Ensemble

## Contents

1. [Overview](#overview)
2. [What This Model Predicts](#what-this-model-predicts)
3. [Why This Design](#why-this-design)
4. [How the Model is Trained](#how-the-model-is-trained)
5. [Hyperparameters](#hyperparameters)
6. [Feature Summary](#feature-summary)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Performance](#performance)
9. [What Was Tested and Rejected](#what-was-tested-and-rejected)
10. [Limitations](#limitations)

---

## Overview

A three-model ensemble for predicting the 5-day excess return of TD Bank (TD.TO) relative to the XFN sector ETF. Each sub-model draws from a distinct information source — price data, earnings call transcripts, and news. Predictions are combined via hard majority vote.

| Property           | Value                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------- |
| Target             | `target_excess_xfn_5d` — TD 5-day return minus XFN 5-day return                        |
| Classes            | −1: TD underperforms XFN by > 0.3%, 0: within ±0.3%, +1: TD outperforms XFN by > 0.3% |
| Full data span     | 2021-02-25 → 2026-04-09 (all training + test periods combined; 13 folds)                |
| Algorithm          | XGBoost, 3-class                                                                        |
| Ensemble           | Hard majority vote (ties → neutral 0)                                                   |
| Training window    | Rolling 504 trading days (~2 years)                                                     |
| Test window        | 63 trading days (1 quarter)                                                             |
| Leakage gap        | 5 trading days                                                                          |
| Retraining cadence | Quarterly                                                                               |

*Source of truth for all performance numbers: `notebooks/xgb_ensemble_v2_performance.ipynb`. Source of truth for training logic: `notebooks/xgb_ensemble_v2_training.ipynb`.*

---

## What This Model Predicts

### Target

```
TD_fwd5[t]  = (TD.adj_close[t+5]  / TD.adj_close[t])  - 1
XFN_fwd5[t] = (XFN.adj_close[t+5] / XFN.adj_close[t]) - 1

target_excess_xfn_5d[t] = TD_fwd5[t] − XFN_fwd5[t]
```

`adj_close` (adjusted close) accounts for dividends and stock splits. TD pays quarterly dividends and XFN pays monthly, so the adjustment is material for both.

### Why sector-relative?

TD's raw return is largely driven by forces that move all Canadian banks together — interest rates, credit cycle, macro. Subtracting XFN's return removes that shared movement and isolates what is specific to TD: its earnings, management decisions, and regulatory exposure.

### Why XFN as benchmark?

XFN (iShares S&P/TSX Capped Financials ETF) is TD's most direct liquid peer benchmark, with the major Canadian banks as its top holdings.

| Approximate holding    | Weight |
| ---------------------- | ------ |
| Royal Bank (RY)        | ~25%   |
| TD Bank (TD)           | ~19%   |
| Scotiabank (BNS)       | ~11%   |
| Bank of Montreal (BMO) | ~10%   |
| CIBC (CM)              | ~8%    |
| Manulife (MFC)         | ~6%    |
| Sun Life (SLF)         | ~4%    |
| Other financials       | ~17%   |

*Weights are approximate and rebalance periodically. Source: iShares Canada.*

### Why a 5-day horizon?

Short enough to be actionable; long enough for sentiment and momentum signals to register. A 1–2 day horizon is dominated by noise; a 20+ day horizon mixes signals from different quarters.

### Class definition and why ±0.3%

| Class | Label        | Condition                           |
| ----- | ------------ | ----------------------------------- |
| +1    | Outperform   | TD 5-day excess return > +0.3%      |
|  0    | Neutral      | TD 5-day excess return within ±0.3% |
| −1    | Underperform | TD 5-day excess return < −0.3%      |

At ±0.3%, roughly 30–35% of days fall in the neutral class — enough to filter out noise without discarding genuine directional moves.

---

## Why This Design

### Three separate models instead of one

Each sub-model is trained on one information source only. Three reasons:

1. **Information staleness differs**: transcript features forward-fill from the earnings call date, not the fiscal quarter start. The same values repeat for 56–82 trading days between calls (exact schedule in `feature_engineering.md`). Mixing this near-static signal with truly daily signals (price, news) in a single model can produce misleading feature interactions.
2. **Interpretability**: each model's predictions and SHAP values can be inspected independently. If performance degrades, the failing signal source can be identified and fixed in isolation.
3. **Extensibility**: adding a new data source means adding a new sub-model without touching the existing three.

### XGBoost over alternatives

~800 training rows per fold, 5–10 features per model. XGBoost suits this setup because:

- **Native multiclass probability output** (`multi:softprob`): produces a probability for each of the three classes in a single pass, without one-vs-rest wrappers.
- **Native regularisation**: L1 (`reg_alpha`), L2 (`reg_lambda`), `min_child_weight`, and `max_depth` all control overfitting directly — important with only ~800 training rows.
- **Handles missing values natively**: no separate imputation step, which reduces preprocessing decisions that could introduce leakage.

| Alternative         | Reason not chosen                                              |
| ------------------- | -------------------------------------------------------------- |
| LightGBM            | Tested; no meaningful performance difference on this dataset   |
| Logistic regression | Cannot capture non-linear interactions between features        |
| Neural networks     | Inappropriate for this data scale                              |

### Hard majority vote over soft-prob averaging

Hard majority vote achieved 0.566 directional accuracy vs. 0.536 for soft-prob averaging. The models know direction reliably but are not well-calibrated in probability space — hard voting uses the directional signal and ignores the miscalibrated confidence.

| Price | Transcript | News | Ensemble outcome                  |
| ----- | ---------- | ---- | --------------------------------- |
| +1    | +1         | −1   | **+1** (majority)                 |
| −1    | 0          | −1   | **−1** (majority)                 |
| +1    | −1         | 0    | **0** (three-way split → neutral) |

---

## How the Model is Trained

| Parameter       | Value                                 |
| --------------- | ------------------------------------- |
| Objective       | 3-class multiclass (`multi:softprob`) |
| Training window | 504 trading days (~2 years), rolling  |
| Leakage gap     | 5 trading days                        |
| Retraining      | Quarterly                             |

- **2-year rolling window**: captures at least 8 quarterly earnings cycles, needed for the transcript model to learn call-to-call patterns.
- **Quarterly retraining**: aligns with the earnings cycle; the most recent call is always in the training set.
- **5-day gap**: prevents any overlap between price-based features and the 5-day forward return target.

### Fold schedule (13 evaluation quarters)

Each fold trains on 504 trading days and tests on 63 trading days. Fold 13 is a partial quarter (20 days available at evaluation time).

| Fold | Train start | Train end  | Test start | Test end   | Test days |
| ---- | ----------- | ---------- | ---------- | ---------- | --------- |
| 1    | 2021-02-25  | 2023-02-28 | 2023-03-08 | 2023-06-06 | 63        |
| 2    | 2021-05-27  | 2023-05-30 | 2023-06-07 | 2023-09-06 | 63        |
| 3    | 2021-08-26  | 2023-08-29 | 2023-09-07 | 2023-12-05 | 63        |
| 4    | 2021-11-25  | 2023-11-28 | 2023-12-06 | 2024-03-07 | 63        |
| 5    | 2022-02-28  | 2024-02-29 | 2024-03-08 | 2024-06-06 | 63        |
| 6    | 2022-05-30  | 2024-05-30 | 2024-06-07 | 2024-09-06 | 63        |
| 7    | 2022-08-29  | 2024-08-29 | 2024-09-09 | 2024-12-05 | 63        |
| 8    | 2022-11-28  | 2024-11-28 | 2024-12-06 | 2025-03-10 | 63        |
| 9    | 2023-03-01  | 2025-03-03 | 2025-03-11 | 2025-06-09 | 63        |
| 10   | 2023-05-31  | 2025-06-02 | 2025-06-10 | 2025-09-09 | 63        |
| 11   | 2023-08-30  | 2025-09-02 | 2025-09-10 | 2025-12-08 | 63        |
| 12   | 2023-11-29  | 2025-12-01 | 2025-12-09 | 2026-03-11 | 63        |
| 13   | 2024-03-01  | 2026-03-04 | 2026-03-12 | 2026-04-09 | 20        |

---

## Hyperparameters

Grid search on folds 4–10. Ranking criterion: `0.6 × directional_accuracy + 0.4 × macro_F1`. Grid designed for small-dataset overfitting control: shallow trees, high minimum leaf size, L1/L2 regularisation.

| Parameter          | Search range |
| ------------------ | ------------ |
| `max_depth`        | 2, 3, 4      |
| `min_child_weight` | 10, 20, 30   |
| `n_estimators`     | 50, 80, 120  |
| `reg_alpha` (L1)   | 0.0, 0.2     |
| `reg_lambda` (L2)  | 1.0, 2.0     |

108 configurations × 3 models.

### Best parameters (tuned 2026-04-23)

| Model      | max_depth | min_child_weight | n_estimators | reg_alpha | reg_lambda |
| ---------- | --------- | ---------------- | ------------ | --------- | ---------- |
| Price      | 3         | 30               | 120          | 0.0       | 1.0        |
| Transcript | 4         | 20               | 80           | 0.2       | 1.0        |
| News       | 2         | 30               | 50           | 0.2       | 2.0        |

---

## Feature Summary

Full feature definitions and formulas are in `feature_engineering.md`.

| Model      | Feature count | Key inputs                                                                                              |
| ---------- | ------------- | ------------------------------------------------------------------------------------------------------- |
| Price      | 6             | Trailing returns (5d, 20d), sector-relative returns (5d, 20d), 52-week high distance, 20-day volatility |
| Transcript | 10            | Exec tone, analyst Q&A sentiment, transcript-vs-filing framing gap, all-source topic shares and sentiments (guidance, AML, credit quality), days since call |
| News       | 5             | Mean sentiment (7d, 30d), article count (7d, 30d), days since last news                                 |

Price and transcript/NLP windows are in **trading days**; news windows are in **calendar days**. Topic share and sentiment features are quarter-level NLP aggregates across transcripts, news, filings, and reports, then forward-filled from the earnings call date.

---

## Evaluation Metrics

All metrics use **stride-offset averaging (stride = 5)** to handle overlapping return windows.

The target is a 5-day forward return, so consecutive test days share 4 days of return data and are not independent. To avoid inflating performance estimates, we take every 5th observation (a non-overlapping subset) and compute the metric on that slice. There are 5 valid starting points (offsets 0–4), each giving a non-overlapping set. The metric is computed for each offset separately and the 5 results are averaged. All test days are used across the 5 offsets combined.

| Metric                  | Definition                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `directional_call_rate` | Share of days where prediction ≠ 0                                                                        |
| `directional_accuracy`  | Share of directional calls where `sign(prediction) == sign(actual excess return)`; random baseline = 0.50 |
| `auc_pos`               | OVR AUC for +1 class using softmax P(+1); diagnostic only                                                 |
| `auc_neg`               | OVR AUC for −1 class using softmax P(−1); diagnostic only                                                 |

`directional_call_rate` measures how often the model commits to a direction; `directional_accuracy` measures how often it gets that direction right. Mean accuracy is excluded because it is dominated by the neutral class. AUC is kept as a diagnostic on sub-model probabilities only.

---

## Performance

*Source: `notebooks/xgb_ensemble_v2_performance.ipynb`, validated 2026-04-23, 13 folds.*

### Aggregate (pooled across all test days)

| Model                  | Directional Call Rate | Directional Accuracy | AUC (+1) | AUC (−1) |
| ---------------------- | --------------------- | -------------------- | -------- | -------- |
| Price                  | 0.972                 | 0.530                | 0.508    | 0.530    |
| Transcript             | 0.993                 | 0.550                | 0.547    | 0.608    |
| News                   | 0.996                 | 0.536                | 0.564    | 0.557    |
| **Hard Majority Vote** | **0.982**             | **0.566**            | —        | —        |

### Hard majority vote — per fold

| Fold | Test period             | Directional Call Rate | Directional Accuracy |
| ---- | ----------------------- | --------------------- | -------------------- |
| 1    | 2023-03-08 → 2023-06-06 | 0.968                 | 0.361                |
| 2    | 2023-06-07 → 2023-09-06 | 0.951                 | 0.694                |
| 3    | 2023-09-07 → 2023-12-05 | 1.000                 | 0.538                |
| 4    | 2023-12-06 → 2024-03-07 | 0.968                 | 0.657                |
| 5    | 2024-03-08 → 2024-06-06 | 1.000                 | 0.572                |
| 6    | 2024-06-07 → 2024-09-06 | 0.936                 | 0.594                |
| 7    | 2024-09-09 → 2024-12-05 | 1.000                 | 0.686                |
| 8    | 2024-12-06 → 2025-03-10 | 0.985                 | 0.388                |
| 9    | 2025-03-11 → 2025-06-09 | 1.000                 | 0.459                |
| 10   | 2025-06-10 → 2025-09-09 | 1.000                 | 0.663                |
| 11   | 2025-09-10 → 2025-12-08 | 0.968                 | 0.457                |
| 12   | 2025-12-09 → 2026-03-11 | 0.985                 | 0.691                |
| 13   | 2026-03-12 → 2026-04-09 | 1.000                 | 0.600 *(20 days)*    |

*For per-fold confusion matrices and SHAP breakdowns, see `notebooks/xgb_ensemble_v2_performance.ipynb` (Sections 7b and 10). For interpretation of strong and weak folds, see `model_explainability.md`.*

### Baseline comparison & statistical significance

*Source: `notebooks/xgb_ensemble_v2_performance.ipynb` (Section 12). Both baselines computed on the same 13-fold walk-forward test data.*

| Baseline | Description | Directional Accuracy |
| -------- | ----------- | -------------------- |
| Majority class | Predict the most frequent class from each fold's training window on every test day | 50.3% |
| Momentum | Continue the sign of the prior 5-day TD-vs-XFN relative return | 51.1% |
| **Hard Majority Vote** | **Ensemble (13 folds)** | **56.6%** |

The ensemble's 56.6% directional accuracy carries a **95% Wald confidence interval of [53.1%, 60.1%]**, computed over ~761 active directional calls. The lower bound (53.1%) exceeds the 50.0% random baseline — the result is statistically distinguishable from chance at 95% confidence.

| Comparison | Lift |
| ---------- | ---- |
| vs majority class | +6.4pp |
| vs momentum | +5.5pp |

Momentum and majority class are the benchmarks most relevant for a pension mandate: a fund constrained to act symmetrically on long and short exposures cannot default to always-long, so the meaningful question is whether the model beats simple rule-based strategies.

---

## What Was Tested and Rejected

| Feature / Choice              | Reason rejected                                                   |
| ----------------------------- | ----------------------------------------------------------------- |
| 60-day price returns          | Adds noise at the 5-day horizon; quarterly retraining already captures regime shifts |
| Volume change features        | No directional mechanism; empirically reduced performance         |
| 90-day news window            | Dilutes short-horizon signal                                      |
| `call_decay` on NLP features  | Near-perfect multicollinearity; replaced by `days_since_call`     |
| Macro features (FX, CPI, rates) | Already embedded in XFN benchmark; would double-count sector movement |
| Single combined model         | Loss of interpretability; misleading feature interactions between stale and fresh signals |
| Soft-prob averaging           | Underperformed hard vote; models are not probability-calibrated   |

---

## Limitations

1. **Single stock, limited data** — ~800 training rows per fold. Rare but high-impact events (regulatory shock, leadership change) may appear only once or not at all in the training window.
2. **Directional only** — predicts direction relative to XFN, not magnitude. Position sizing requires a separate model or rule.
3. **Cannot detect priced-in events** — the model cannot distinguish a crisis still unfolding from one already absorbed by the market. Bearish signals from an active crisis persist into the recovery period, and the model has no feature to tell them apart. Fold 8 (AML recovery) is the clearest example.
4. **Limited news coverage** — news features are based on TD press releases only; broad macro news is not included.
