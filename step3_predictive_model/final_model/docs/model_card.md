# Model Card — Three-Model Ensemble

## Overview

A three-model ensemble for predicting 5-day excess return of TD Bank (TD.TO) relative to the XFN sector ETF. Each sub-model draws from a distinct, non-overlapping information source — price/market data, earnings call transcripts, and news media. Sub-model predictions are combined via hard majority vote.


| Property           | Value                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------- |
| Target             | `target_excess_xfn_5d` — TD 5-day return minus XFN 5-day return                       |
| Classes            | −1: TD underperforms XFN by > 0.3%, 0: within ±0.3%, +1: TD outperforms XFN by > 0.3% |
| Algorithm          | XGBoost, 3-class                                                                      |
| Ensemble           | Hard majority vote (ties → neutral 0)                                                 |
| Training window    | Rolling 504 trading days (~2 years)                                                   |
| Test window        | 63 trading days (1 quarter)                                                           |
| Leakage gap        | 5 trading days                                                                        |
| Retraining cadence | Quarterly                                                                             |


---

## Target Definition

```
TD_fwd5[t]  = (TD.adj_close[t+5]  / TD.adj_close[t])  - 1   # forward 5-day return
XFN_fwd5[t] = (XFN.adj_close[t+5] / XFN.adj_close[t]) - 1   # forward 5-day return

target_excess_xfn_5d[t] = TD_fwd5[t] − XFN_fwd5[t]          # percentage-point difference
```

`adj_close` (adjusted close) accounts for dividends and stock splits by back-adjusting historical prices. Using raw `close` would introduce artificial return gaps on ex-dividend dates, distorting both the target and any price-based features. Both TD and XFN pay regular dividends — TD quarterly, XFN monthly — so the adjustment is material. Both series use `adj_close` consistently.

*For the rationale behind benchmark selection and target design, see `model_choice.md`.*

---

## Feature Definitions

*Full feature definitions, formulas, and design rationale are documented in `feature_engineering.md`.*

### Summary

| Model | Feature count | Key inputs |
|---|---|---|
| Price | 6 | Trailing returns (5d, 20d), sector-relative returns, 52-week distance, 20-day volatility |
| Transcript | 10 | Exec tone, analyst Q&A sentiment, framing gap, topic shares and sentiments (guidance, AML, credit quality), days since call |
| News | 5 | Mean sentiment (7d, 30d), article count (7d, 30d), days since last news |

Price and transcript windows are in **trading days**; news windows and `days_since_last_news` are in **calendar days**.

---

## Training Methodology


| Parameter       | Value                                 |
| --------------- | ------------------------------------- |
| Objective       | 3-class multiclass (`multi:softprob`) |
| Training window | 504 trading days, rolling             |
| Gap             | 5 trading days (leakage prevention)   |
| Retraining      | Quarterly                             |


---

## Hyperparameter Search

Grid searched on folds 4–10 (7 middle folds of 13). Ranking criterion: composite score `0.6 × directional_accuracy + 0.4 × macro_F1` (tuning only; not a reported metric).


| Parameter          | Search range |
| ------------------ | ------------ |
| `max_depth`        | 2, 3, 4      |
| `min_child_weight` | 10, 20, 30   |
| `n_estimators`     | 50, 80, 120  |
| `reg_alpha` (L1)   | 0.0, 0.2     |
| `reg_lambda` (L2)  | 1.0, 2.0     |


Total: 108 configurations × 3 models.

### Best parameters (tuned 2026-04-23)


| Model      | max_depth | min_child_weight | n_estimators | reg_alpha | reg_lambda |
| ---------- | --------- | ---------------- | ------------ | --------- | ---------- |
| Price      | 3         | 30               | 120          | 0.0       | 1.0        |
| Transcript | 4         | 20               | 80           | 0.2       | 1.0        |
| News       | 2         | 30               | 50           | 0.2       | 2.0        |


---

## Ensemble Method


| Price | Transcript | News | Ensemble                          |
| ----- | ---------- | ---- | --------------------------------- |
| +1    | +1         | −1   | **+1** (majority)                 |
| −1    | 0          | −1   | **−1** (majority)                 |
| +1    | −1         | 0    | **0** (three-way split → neutral) |


Directional agreement across models is more reliable than probability calibration on this dataset — hard majority vote outperforms soft-prob averaging, indicating the models share strong directional signal but are not well-calibrated in probability space.

---

## Evaluation Metrics

All metrics use stride-offset averaging (stride = 5) to avoid overlap from the 5-day forward return target.


| Metric                  | Definition                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `directional_call_rate` | Share of days where prediction != 0                                                                        |
| `directional_accuracy`  | Share of directional calls where `sign(prediction) == sign(actual excess return)`; random baseline = 0.50 |
| `auc_pos`               | OVR AUC for +1 class using softmax P(+1); diagnostic only                                                 |
| `auc_neg`               | OVR AUC for −1 class using softmax P(−1); diagnostic only                                                 |


**Why these four?** This is a single-stock, directional-signal model — the goal is to know *when* to take a position and *which direction*, not to rank a universe. `directional_call_rate` measures selectivity (how often the model commits); `directional_accuracy` measures quality of those commitments. Together they capture the practical value of the signal. Mean accuracy is excluded because it is dominated by the neutral class and gives a misleading picture of directional skill. Macro F1 is excluded for the same reason. AUC (+1/−1) are kept as diagnostic probes on the sub-models' probability outputs — useful for understanding calibration but not the primary decision metric since the ensemble uses hard votes, not probabilities.

---

## Performance (validated 2026-04-23, 13 folds)

### Aggregate


| Model                  | Directional Call Rate | Directional Accuracy | AUC (+1) | AUC (−1) |
| ---------------------- | --------------------- | -------------------- | -------- | -------- |
| Price                  | 0.972                 | 0.530                | 0.508    | 0.530    |
| Transcript             | 0.993                 | 0.550                | 0.547    | 0.608    |
| News                   | 0.996                 | 0.536                | 0.564    | 0.557    |
| **Hard Majority Vote** | **0.982**             | **0.566**            | —        | —        |


### Hard majority vote — per fold


| Fold     | Directional Call Rate | Directional Accuracy |
| -------- | --------------------- | -------------------- |
| 1        | 0.968                 | 0.361                |
| 2        | 0.951                 | 0.694                |
| 3        | 1.000                 | 0.538                |
| 4        | 0.968                 | 0.657                |
| 5        | 1.000                 | 0.572                |
| 6        | 0.936                 | 0.594                |
| 7        | 1.000                 | 0.686                |
| 8        | 0.985                 | 0.388                |
| 9        | 1.000                 | 0.459                |
| 10       | 1.000                 | 0.663                |
| 11       | 0.968                 | 0.457                |
| 12       | 0.985                 | 0.691                |
| 13       | 1.000                 | 0.600                |
| **Mean** | **0.982**             | **0.566**            |


*For per-fold and per-model SHAP analysis, see `model_explainability.md`.*

---

## Limitations

1. **Single stock, limited data** — training on one stock produces ~800 rows per fold. This constrains model complexity and limits the model's exposure to rare but high-impact events (credit stress, regulatory shock, leadership change).
2. **Directional only** — predicts direction relative to XFN, not magnitude. Position sizing requires a separate model.
3. **Lack of clean structured LLM data** — news coverage is limited to TD's own press releases; broad macro news is absent. Transcript features update quarterly and are forward-filled between calls, meaning they can be significantly stale in the weeks before the next earnings event. Both gaps reflect the absence of a licensed, structured news feed with consistent daily coverage.

