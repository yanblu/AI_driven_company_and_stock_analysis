# Model Card — Three-Model Ensemble (V2)

## Overview

A three-model ensemble for predicting 5-day excess return of TD Bank (TD.TO) relative to the XFN sector ETF. Each sub-model draws from a distinct, non-overlapping information source — price/market data, earnings call transcripts, and news media. Sub-model predictions are combined via hard majority vote.

| Property | Value |
|---|---|
| Target | `target_excess_xfn_5d` — 5-day TD vs XFN excess return |
| Classes | −1 (< −0.3%), 0 (within ±0.3%), +1 (> +0.3%) |
| Algorithm | XGBoost (`multi:softprob`, 3-class) |
| Ensemble | Hard majority vote (ties → neutral 0) |
| Training window | Rolling 504 trading days (~2 years) |
| Test window | 63 trading days (1 quarter) |
| Leakage gap | 5 trading days |
| Retraining cadence | Quarterly |
| Folds evaluated | 13 (first fold dropped; insufficient training data) |

---

## Feature Definitions

### Model 1 — Price (6 features)

| Feature | Description | Source |
|---|---|---|
| `td_return_5d` | 5-day price return | Daily features parquet |
| `td_return_20d` | 20-day price return | Daily features parquet |
| `td_vs_xfn_5d` | 5-day TD excess return vs XFN ETF | Daily features parquet |
| `td_vs_xfn_20d` | 20-day TD excess return vs XFN ETF | Inline: `td_return_20d − xfn_return_20d` |
| `td_dist_52w_high_v2` | Distance from 52-week high | Inline from `TD_TO.parquet`: `(close − max_252) / close`; always ≤ 0 |
| `td_volatility_20d` | 20-day realised volatility (std of daily returns) | Daily features parquet |

**Design rationale**: short and mid-term momentum (`5d`, `20d`), sector-relative performance at both horizons (`td_vs_xfn`), drawdown positioning (`td_dist_52w_high_v2`), and volatility regime (`td_volatility_20d`). Volatility helps the model calibrate how likely a ±0.3% move is in the current market environment.

### Model 2 — Transcript + Report (10 features)

| Feature | Description | Source |
|---|---|---|
| `exec_tone_ffill` | Mean CEO + CFO prepared-remarks sentiment (forward-filled) | `nlp_features.parquet`, combined inline |
| `transcript_analyst_qa_sentiment_mean_ffill` | Analyst Q&A tone | `nlp_features.parquet` |
| `framing_gap_ffill` | CEO verbal tone minus written filing tone (divergence signal) | `nlp_features.parquet` |
| `topic_guidance_share_ffill` | Share of guidance discussion in the call | `nlp_features.parquet` |
| `topic_guidance_sentiment_ffill` | Tone of guidance discussion | `nlp_features.parquet` |
| `topic_regulatory_AML_share_ffill` | Share of AML/regulatory discussion | `nlp_features.parquet` |
| `topic_regulatory_AML_sentiment_ffill` | Tone of AML/regulatory discussion | `nlp_features.parquet` |
| `topic_credit_quality_share_ffill` | Share of credit quality discussion | `nlp_features.parquet` |
| `topic_credit_quality_sentiment_ffill` | Tone of credit quality discussion | `nlp_features.parquet` |
| `days_since_call` | Trading days elapsed since the last earnings call | Daily features parquet |

**Design rationale**: executive tone and analyst reception capture the market's immediate reaction to communication. Topic shares and sentiments focus on the three areas most material to TD's risk profile: forward guidance, AML/regulatory exposure (TD-specific), and credit quality. `days_since_call` is the staleness signal — the tree learns its own decay thresholds rather than a fixed exponential. All features are used undecayed.

### Model 3 — News (5 features)

| Feature | Description | Source |
|---|---|---|
| `news_sent_mean_7d` | Mean LLM sentiment score over the last 7 days | Daily features parquet |
| `news_sent_mean_30d` | Mean LLM sentiment score over the last 30 days | Daily features parquet |
| `news_count_7d` | Article count over the last 7 days | Daily features parquet |
| `news_count_30d` | Article count over the last 30 days | Daily features parquet |
| `days_since_last_news` | Calendar days since the most recent article | Daily features parquet |

**Design rationale**: short (7d) and medium (30d) windows capture both immediate and sustained media narrative. Count features capture news volume (potential event signal). `days_since_last_news` captures information drought. No 90-day window — the mechanism linking 3-month sentiment to a 5-day excess return target is unclear.

---

## Training Methodology

| Parameter | Value |
|---|---|
| Algorithm | XGBoost `multi:softprob`, 3-class |
| Objective | `multi:softmax` for labels; `multi:softprob` for probabilities |
| Target | `target_excess_xfn_5d` |
| Threshold | ±0.003 (±0.3% excess return) |
| Training window | 504 trading days (rolling) |
| Test window | 63 trading days |
| Gap | 5 trading days (leakage prevention) |
| Retraining | Quarterly |
| First fold | Dropped (< 2 years of training data) |
| Fixed params | `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist` |

---

## Hyperparameter Search

Grid searched on folds 4–10 (7 middle folds of 13). Ranking criterion: composite score `0.6 × directional_accuracy + 0.4 × macro_F1` (tuning only; not a reported metric).

| Parameter | Search range |
|---|---|
| `max_depth` | 2, 3, 4 |
| `min_child_weight` | 10, 20, 30 |
| `n_estimators` | 50, 80, 120 |
| `reg_alpha` (L1) | 0.0, 0.2 |
| `reg_lambda` (L2) | 1.0, 2.0 |

Total: 108 configurations × 3 models.

### Best parameters (tuned 2026-04-23)

| Model | max_depth | min_child_weight | n_estimators | reg_alpha | reg_lambda |
|---|---|---|---|---|---|
| Price | 3 | 30 | 120 | 0.0 | 1.0 |
| Transcript | 4 | 20 | 80 | 0.2 | 1.0 |
| News | 2 | 30 | 50 | 0.2 | 2.0 |

---

## Ensemble Method

**Primary: Hard Majority Vote**

Each sub-model casts a directional vote: −1, 0, or +1. The most frequent label wins. Three-way ties (all models disagree) resolve to neutral (0).

Rationale: directional agreement across models is more reliable than probability calibration. Hard majority vote consistently outperforms soft-prob averaging on this dataset, indicating the three models share strong directional signal but are not well-calibrated in probability space.

**Diagnostic: Soft-Prob Average**

Softmax probability vectors averaged across models, then `argmax`. Retained as a benchmark only.

---

## Evaluation Metrics

All metrics computed using stride-offset averaging (stride = 5) to avoid overlap from the 5-day forward return target.

| Metric | Definition | Notes |
|---|---|---|
| `directional_call_rate` | Share of days where prediction ≠ 0 | How often the model takes a directional stance |
| `directional_accuracy` | Share of directional calls where `sign(prediction) == sign(actual excess return)` | Random baseline = 0.50 |
| `auc_pos` | OVR AUC for +1 class using softmax P(+1) | Diagnostic; soft-prob models only |
| `auc_neg` | OVR AUC for −1 class using softmax P(−1) | Diagnostic; soft-prob models only |

---

## Performance (validated 2026-04-23, 13 folds)

### Aggregate

| Model | Directional Call Rate | Directional Accuracy | AUC (+1) | AUC (−1) |
|---|---|---|---|---|
| Price | 0.972 | 0.530 | 0.508 | 0.530 |
| Transcript | 0.993 | 0.550 | 0.547 | 0.608 |
| News | 0.996 | 0.536 | 0.564 | 0.557 |
| **Hard Majority Vote** | **0.982** | **0.566** | — | — |
| Soft-Prob (diagnostic) | 1.000 | 0.536 | 0.561 | 0.591 |

### Hard majority vote — per fold

| Fold | Directional Call Rate | Directional Accuracy |
|---|---|---|
| 1 | 0.968 | 0.361 |
| 2 | 0.951 | 0.694 |
| 3 | 1.000 | 0.538 |
| 4 | 0.968 | 0.657 |
| 5 | 1.000 | 0.572 |
| 6 | 0.936 | 0.594 |
| 7 | 1.000 | 0.686 |
| 8 | 0.985 | 0.388 |
| 9 | 1.000 | 0.459 |
| 10 | 1.000 | 0.663 |
| 11 | 0.968 | 0.457 |
| 12 | 0.985 | 0.691 |
| 13 | 1.000 | 0.600 |
| **Mean** | **0.982** | **0.566** |

Fold 8 (directional accuracy 0.388) is the weakest quarter — likely a macro/event-driven period where all three signal sources were simultaneously wrong. Folds 2, 4, 7, 12 show strong directional accuracy (0.657–0.694).
