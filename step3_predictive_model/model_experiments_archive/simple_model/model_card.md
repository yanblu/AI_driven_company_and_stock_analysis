# Simple Model Card — lgbm_3class_xfn5d_simple

## Overview

| Property | Value |
|---|---|
| Model ID | `lgbm_3class_xfn5d_simple` |
| Model family | LightGBM multiclass classifier |
| Target | `target_excess_xfn_5d` — 5-day excess return of TD vs XFN sector ETF |
| Labels | Outperform (+1) / Neutral (0) / Underperform (−1) |
| Threshold | ±0.30% (±30 bps) |
| Feature count | **15** (raw, explainable only — no technical indicators, no macro/global risk) |
| Validation | 7-fold expanding-window walk-forward (stride-5 offset averaging) |
| Baseline for comparison | `lgbm_3class_xfn5d_35f` (35 features, final model) |

---

## Best Hyperparameters

Selected via 36-config × 7-fold walk-forward grid search. Criterion: composite = 0.6 × ASA + 0.4 × Macro F1.

| Parameter | Simple 15f | Final 35f |
|---|---|---|
| `num_leaves` | 4 | 4 |
| `min_child_samples` | 40 | 40 |
| `n_estimators` | 200 | 80 |
| `learning_rate` | 0.05 | 0.05 |
| `feature_fraction` | 0.8 | 0.8 |
| `reg_alpha` | 0.2 | 0.2 |
| `reg_lambda` | 1.0 | 1.0 |

---

## Performance — 7-Fold Walk-Forward

### Aggregate (mean across 7 folds)

| Metric | Simple 15f | Final 35f | Delta |
|---|---|---|---|
| Active Sign Accuracy (ASA) | **0.542** | 0.605 | −0.063 |
| Macro F1 | **0.325** | 0.347 | −0.022 |
| Dir AUC | **0.528** | 0.572 | −0.044 |
| Active coverage | 0.931 | 0.968 | −0.037 |
| Mean accuracy | 0.414 | 0.482 | −0.068 |
| Composite (0.6×ASA + 0.4×F1) | **0.455** | 0.502 | −0.047 |

### Per-Fold Results — Simple 15f

| Fold | Test period | Acc | ASA | Coverage | Dir AUC |
|---|---|---|---|---|---|
| v1 | 2023-01-10 → 2023-06-30 | 0.313 | 0.517 | 0.843 | 0.456 |
| v2 | 2023-07-11 → 2023-12-29 | 0.411 | 0.576 | 0.932 | 0.543 |
| v3 | 2024-01-09 → 2024-06-28 | 0.555 | 0.679 | 0.975 | 0.612 |
| v4 | 2024-07-09 → 2024-12-31 | 0.487 | 0.559 | 0.975 | 0.624 |
| v5 | 2025-01-09 → 2025-06-30 | 0.325 | 0.412 | 0.892 | 0.391 |
| v6 | 2025-07-09 → 2025-12-31 | 0.405 | 0.550 | 0.918 | 0.598 |
| v7 | 2026-01-09 → 2026-04-09 | 0.399 | 0.500 | 0.985 | 0.473 |
| **Mean** | | **0.414** | **0.542** | **0.931** | **0.528** |

### Per-Fold Results — Final 35f (for reference)

| Fold | Acc | ASA | Coverage | Dir AUC |
|---|---|---|---|---|
| v1 | 0.504 | 0.647 | 0.909 | 0.634 |
| v2 | 0.470 | 0.577 | 0.958 | 0.599 |
| v3 | 0.479 | 0.595 | 1.000 | 0.469 |
| v4 | 0.529 | 0.611 | 1.000 | 0.640 |
| v5 | 0.400 | 0.511 | 0.908 | 0.532 |
| v6 | 0.528 | 0.676 | 1.000 | 0.626 |
| v7 | 0.463 | 0.619 | 1.000 | 0.503 |
| **Mean** | **0.482** | **0.605** | **0.968** | **0.572** |

---

## Same 15 features — XGBoost vs LightGBM

Trained and tuned in [`xgb_3class_xfn5d_simple_training.ipynb`](xgb_3class_xfn5d_simple_training.ipynb): XGBoost uses a separate grid (`max_depth`, `min_child_weight`, `n_estimators`); LightGBM uses the saved `lgbm_3class_xfn5d_simple_tuned-*.pkl` `best_params`. Both are re-fit per fold on the same walk-forward splits (no look-ahead).

**Best XGBoost search winner** (7-fold composite): `max_depth=3`, `min_child_weight=1`, `n_estimators=80` — composite **0.444** vs LightGBM simple search composite **0.455**.

### 7-fold mean — walk-forward re-fit (Section 6 of XGB notebook)

| Metric | XGB 15f | LGBM 15f | Δ (XGB − LGBM) |
|---|---|---|---|
| Mean accuracy | 0.415 | 0.414 | +0.002 |
| Macro F1 | 0.310 | 0.325 | −0.015 |
| Active coverage | 0.976 | 0.931 | +0.045 |
| Active sign accuracy (ASA) | 0.533 | 0.542 | −0.009 |
| Composite (0.6×ASA + 0.4×F1) | **0.444** | **0.455** | **−0.011** |

**Takeaway**: On this dataset and feature set, LightGBM slightly wins on ASA and F1, so the composite favors LGBM. XGBoost issues more active (non-neutral) predictions on average (higher coverage), but that does not translate into better directional accuracy on the continuous excess return for active days.

---

## SHAP Feature Importance (Simple 15f)

Mean |SHAP| across 7 folds (all classes averaged):

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | `news_sent_mean_30d` | 0.1481 | News |
| 2 | `evt_aml_sentiment` | 0.1407 | NLP |
| 3 | `evt_framing_gap` | 0.1244 | NLP |
| 4 | `td_return_20d` | 0.1231 | Price |
| 5 | `evt_guidance_topic` | 0.1133 | NLP |
| 6 | `td_dist_52w_high` | 0.1046 | Price |
| 7 | `td_vs_xfn_5d` | 0.1027 | Price |
| 8 | `news_count_30d` | 0.0904 | News |
| 9 | `evt_exec_tone` | 0.0864 | NLP |
| 10 | `evt_aml_topic` | 0.0813 | NLP |
| 11 | `evt_guidance_sentiment` | 0.0541 | NLP |
| 12 | `td_return_5d` | 0.0499 | Price |
| 13 | `days_since_call` | 0.0473 | Timing |
| 14 | `days_since_last_news` | 0.0274 | News |
| 15 | `is_earnings_week` | 0.0000 | Timing |

**Observations**:
- News sentiment (`news_sent_mean_30d`) becomes the #1 feature in the simple model, reflecting its strong raw signal when the top technical indicators are removed.
- AML and guidance NLP features dominate positions 2–5, confirming the importance of the earnings call narrative even in simplified form.
- Price features (`td_return_20d`, `td_dist_52w_high`, `td_vs_xfn_5d`) contribute meaningfully but are no longer the dominant drivers.
- `is_earnings_week` has zero SHAP — the model learns earnings timing from `days_since_call` instead.

---

## Feature Set

### Price & Relative Momentum (4)
- `td_return_5d` — 5-day TD return
- `td_return_20d` — 20-day TD return
- `td_vs_xfn_5d` — 5-day TD return minus XFN return
- `td_dist_52w_high` — distance from 52-week high (mean-reversion anchor)

### News (3)
- `news_sent_mean_30d` — 30-day rolling average news sentiment
- `news_count_30d` — 30-day rolling news article count
- `days_since_last_news` — recency of news

### Timing (2)
- `days_since_call` — days since last earnings call
- `is_earnings_week` — binary flag for earnings announcement week

### NLP Event Signals (6, all × call_decay)
- `evt_exec_tone` — mean(CEO, CFO) prepared-remarks sentiment × call_decay *(new)*
- `evt_framing_gap` — gap between management tone and filed reports × call_decay
- `evt_guidance_topic` — guidance topic share × call_decay *(simplified)*
- `evt_guidance_sentiment` — guidance sentiment × call_decay *(simplified)*
- `evt_aml_topic` — AML regulatory topic share × call_decay *(simplified)*
- `evt_aml_sentiment` — AML regulatory sentiment × call_decay *(simplified)*

---

## Known Trade-offs vs Final 35f Model

| Trade-off | Impact |
|---|---|
| Dropped `td_rsqr_60d` (was #1 SHAP in 35f) | ~3% ASA loss |
| Dropped `td_corr_pv_20d` (was #2 SHAP in 35f) | Additional ASA loss |
| No explicit guidance shift signal (`evt_guidance_shift` dropped) | Model must infer shift from raw components |
| No macro risk context (VIX, yield curve, DXY dropped) | Reduced regime awareness |
| Simplified AML/guidance features (no combined pressure/shift signals) | Loss of pre-engineered interaction |

---

## Artifacts

| File | Description |
|---|---|
| `artifacts/lgbm_3class_xfn5d_simple_tuned-YYYY-MM-DD.pkl` | Serialised LightGBM + feature list + params |
| `artifacts/lgbm_3class_xfn5d_simple_tuned-YYYY-MM-DD.json` | JSON metadata |
| `artifacts/xgb_3class_xfn5d_simple_tuned-YYYY-MM-DD.pkl` | Serialised XGBoost + same 15 features + params |
| `artifacts/xgb_3class_xfn5d_simple_tuned-YYYY-MM-DD.json` | XGBoost JSON metadata |
| `lgbm_3class_xfn5d_simple_training.ipynb` | LightGBM hyperparameter search + training |
| `lgbm_3class_xfn5d_simple_performance_and_shap.ipynb` | Walk-forward evaluation + SHAP (mirrors 35f performance notebook) |
| `xgb_3class_xfn5d_simple_training.ipynb` | XGBoost tuning, artifact save, LGBM vs XGB walk-forward summary |
| `feature_engineering.md` | Full feature audit and construction details |
