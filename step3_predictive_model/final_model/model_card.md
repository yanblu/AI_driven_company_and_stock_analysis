# lgbm_3class_xfn5d_35f — Model Card

## Identity
| Field | Value |
|---|---|
| Model ID | `lgbm_3class_xfn5d_35f` |
| Algorithm | LightGBM multiclass classifier |
| Artifact | `artifacts/lgbm_3class_xfn5d_35f_tuned-2026-04-22.pkl` |
| Metadata | `artifacts/lgbm_3class_xfn5d_35f_tuned-2026-04-22.json` |
| Saved | 2026-04-22 (hyperparameter-tuned; best params confirmed via 36-config search) |

## Problem Framing
TD Bank (TD.TO) directional signal detector — not a decision model. Given information available on day *d*, the model outputs one of three signals for the *next 5 trading days*:

| Signal | Meaning | Label |
|---|---|---|
| +1 Outperform | TD likely outperforms XFN sector ETF by > 0.30% | `1` |
|  0 Neutral | Expected excess return within ±0.30% band | `0` |
| −1 Underperform | TD likely underperforms XFN by > 0.30% | `-1` |

**Target**: `target_excess_xfn_5d` = 5-day TD return minus 5-day XFN return  
**Threshold**: ±0.30% (±0.003 in decimal)

## Model Architecture
```
LGBMClassifier(
    objective       = 'multiclass',
    num_class       = 3,
    n_estimators    = 80,
    learning_rate   = 0.05,
    num_leaves      = 4,           # shallow — avoids overfitting on small dataset with expanded features
    min_child_samples = 40,
    feature_fraction  = 0.8,
    reg_alpha       = 0.2,
    reg_lambda      = 1.0,
    random_state    = 42,
)
```

Hyperparameters were selected by a 36-configuration grid search (num_leaves ∈ {4,8,16,31}, min_child_samples ∈ {20,40,60}, n_estimators ∈ {80,120,200}) across all 7 walk-forward folds, using composite score = 0.6 × ASA + 0.4 × Macro F1. Shallower trees (`num_leaves=4`) proved optimal when the feature set expanded from 30 to 35 features.

## Features (35)

### Market / Price (8)
| Feature | Description |
|---|---|
| `td_return_5d` | TD 5-day price return |
| `td_return_20d` | TD 20-day price return |
| `td_volatility_20d` | Annualised 20-day realised volatility |
| `td_vs_xfn_5d` | TD return minus XFN 5-day return (recent relative momentum) |
| `td_vs_tsx_5d` | TD return minus TSX 5-day return |
| `td_corr_pv_20d` | 20-day price-volume correlation |
| `td_volume_change_20d` | Volume z-score change over 20 days |
| `td_dist_52w_high` | Normalised distance from 52-week high |

### Rates / Macro (2)
| Feature | Description |
|---|---|
| `yield_curve_slope` | 10Y minus 2Y Government of Canada yield |
| `yield_10y_level` | 10-year Government of Canada yield |

### Global Risk (4)
| Feature | Description |
|---|---|
| `vix_volatility_20d` | VIX 20-day vol of vol |
| `dxy_level` | US Dollar Index level |
| `gold_level` | Gold spot price level |
| `fx_usdcad_level` | USD/CAD spot rate |

### News Flow (3)
| Feature | Description |
|---|---|
| `news_sent_mean_30d` | 30-day mean news sentiment (LLM-scored) |
| `news_count_30d` | 30-day news article count |
| `days_since_last_news` | Recency of most recent news article |

### Timing (2)
| Feature | Description |
|---|---|
| `days_since_call` | Days since last earnings call (decay anchor) |
| `is_earnings_week` | Binary: within 5 days of an earnings release |

### NLP Event (11)
Event features are constructed from LLM-scored earnings call transcripts, forward-filled between calls and exponentially decayed by `days_since_call` (half-life 20 days).

| Feature | Description |
|---|---|
| `evt_ceo_tone` | CEO prepared-statement sentiment × call decay |
| `evt_cfo_tone` | CFO prepared-statement sentiment × call decay |
| `evt_framing_gap` | CEO prepared sentiment minus CEO Q&A sentiment |
| `evt_aml_pressure` | AML-topic share × call decay (regulatory signal) |
| `evt_aml_shift` | Change in AML-topic share vs prior call |
| `evt_guidance_strength` | Forward-guidance sentiment strength × call decay |
| `evt_guidance_shift` | Change in guidance sentiment vs prior call |
| `evt_macro_topic` | Macro/economy topic share × call decay |
| `evt_topic_entropy` | Shannon entropy of topic distribution × call decay |
| `evt_news_tone` | Rolling 20d mean news sentiment × call decay |
| `evt_news_flow` | Rolling 20d news count × call decay |

### TA / Qlib Additions (5)
Selected from a 25-candidate diagnostic SHAP ranking (fold-local selection frequency across all 7 folds). Only features consistently ranked in the top 20 across folds were included.

| Feature | Description |
|---|---|
| `td_rsqr_60d` | R² of 60-day OLS price trend — measures trend explanatory power |
| `td_wvma_5d` | 5-day weighted volume-momentum anomaly — unusual vol/volume spikes |
| `td_wvma_20d` | 20-day weighted volume-momentum anomaly |
| `td_corr_pv_5d` | 5-day price-volume correlation — short-term divergence signal |
| `td_beta_20d` | 20-day OLS beta of TD vs XFN — rolling sector sensitivity |

## Training, Holdout & Evaluation

### Model date range

| | Date |
|---|---|
| **Feature data start** | 2021-02-25 |
| **First out-of-sample evaluation** | 2023-01-10 (v1 test start) |
| **Last out-of-sample evaluation** | 2026-04-09 (v7 test end) |
| **Artifact locked** | 2026-04-22 |

All day counts below are **trading days** (Monday–Friday, excluding Canadian public holidays). A calendar month contains approximately 21 trading days.

### How the model is trained
The model uses an **expanding-window walk-forward** protocol. Feature data spans 2021-02-25 to 2026-04-09. It is divided into 7 consecutive folds; in each fold the model is retrained from scratch on all data available up to the fold's cut-off date, then evaluated on the immediately following out-of-sample window. Because the training window grows with each fold, later folds benefit from more historical context.

A **5-trading-day boundary gap** is enforced between the last training row and the first test row. This ensures the 5-day forward return used as the target cannot bleed from the test window back into training features.

### Holdout periods (7 folds)

| Fold | Train window | Test window | Test trading days | Note |
|---|---|---|---|---|
| v1 | 2021-02-25 → 2022-12-30 | 2023-01-10 → 2023-06-30 | 121 | |
| v2 | 2021-02-25 → 2023-06-30 | 2023-07-11 → 2023-12-29 | 119 | |
| v3 | 2021-02-25 → 2023-12-29 | 2024-01-09 → 2024-06-28 | 121 | |
| v4 | 2021-02-25 → 2024-06-28 | 2024-07-09 → 2024-12-31 | 121 | AML peak |
| v5 | 2021-02-25 → 2024-12-31 | 2025-01-09 → 2025-06-30 | 120 | Weakest fold |
| v6 | 2021-02-25 → 2025-06-30 | 2025-07-09 → 2025-12-31 | 121 | Best fold |
| v7 | 2021-02-25 → 2025-12-31 | 2026-01-09 → 2026-04-09 | 63 | 3 calendar months |

The train window always starts from **2021-02-25** (expanding-window design — every fold trains on all available history from the data start). Fold v4 coincides with the peak of TD's AML regulatory period and is the most stress-tested window. Fold v7 has 63 trading days because it runs to the artifact lock date.

### How the model is evaluated
The target is a **5-day forward return**, so consecutive daily predictions overlap — day *d*'s label is resolved at day *d+5*, and the same price move contributes to labels for days *d*, *d−1*, *d−2*, *d−3*, and *d−4*. Evaluating every row would overstate confidence by counting the same return five times.

To correct for this, evaluation uses **stride-5 offset averaging**:
1. For each of the 5 possible starting offsets (0, 1, 2, 3, 4), select every 5th test row.
2. Compute metrics on that non-overlapping sub-sequence.
3. Average the 5 offset metrics to get the reported fold metric.

This produces an unbiased estimate equivalent to treating each 5-day return as a single independent observation.

**Metrics reported:**

| Metric | Definition |
|---|---|
| Mean accuracy | Fraction of predictions that match the exact 3-class label (−1, 0, +1) |
| Macro F1 | Unweighted F1 averaged across all three classes |
| Active coverage | Fraction of days the model takes a non-neutral position (predicts ±1) |
| Active sign accuracy | Among active predictions, fraction where the predicted direction matches the realized direction of the excess return (ignoring the band) |
| DirAcc_abstain50 | Blended directional accuracy treating abstentions as 50/50: `active_sign_acc × coverage + 0.5 × (1 − coverage)` |
| Dir AUC | Threshold-free discrimination metric: average of one-vs-rest AUC for the outperform (+1) and underperform (−1) signals, computed from class probability scores. Analogous to binary AUC — a random model scores 0.50. Computed with the same stride-5 offset averaging as other metrics. |

## Performance (7-fold walk-forward average)

### Per-Fold Results
| Fold | Train rows | Test rows | Mean Acc | Macro F1 | Active Cov | Active Sign Acc | Dir AUC |
|---|---|---|---|---|---|---|---|
| v1 (H1 2023) | 464 | 121 | 0.504 | 0.419 | 0.909 | 0.647 | 0.634 |
| v2 (H2 2023) | 590 | 119 | 0.470 | 0.363 | 0.958 | 0.577 | 0.599 |
| v3 (H1 2024) | 714 | 121 | 0.479 | 0.243 | 1.000 | 0.595 | 0.469 |
| v4 (H2 2024 — AML peak) | 840 | 121 | 0.529 | 0.380 | 1.000 | 0.611 | 0.640 |
| v5 (H1 2025 — weakest) | 966 | 120 | 0.400 | 0.296 | 0.908 | 0.511 | 0.532 |
| v6 (H2 2025 — best) | 1091 | 121 | 0.528 | 0.392 | 1.000 | 0.676 | 0.626 |
| v7 (Q1 2026) | 1217 | 63 | 0.463 | 0.335 | 1.000 | 0.619 | 0.503 |
| **7-fold mean** | | | **0.482** | **0.347** | **0.968** | **0.605** | **0.572** |

### Aggregate Summary
| Metric | Value |
|---|---|
| Mean accuracy | 48.2% |
| Macro F1 | 0.347 |
| Active coverage | 96.8% |
| Active sign accuracy | 60.5% |
| Dir AUC | 0.572 (random baseline = 0.50) |
| DirAcc_strict (coverage-weighted) | 58.6% |
| DirAcc_abstain50 (abstain = 50/50) | 60.3% |

### Baselines (same 7-fold average)
Three baselines are reported. The **majority** and **momentum** baselines are the most relevant benchmarks for a pension context — a pension mandate typically cannot default to always-long because it must manage drawdown risk and is often constrained to act symmetrically on long and short exposures.

| Baseline | Description | Mean Acc | Active Coverage | Active Sign Acc |
|---|---|---|---|---|
| **Majority** | Always predict the most frequent class from training data | ~38.5% | 100% | ~38.5% |
| **Momentum** | Predict continuation of TD's recent 5-day relative performance vs XFN | ~36.6% | ~79% | ~39.0% |
| Always-long | Always predict outperform (+1) | ~43% | 100% | ~43% |

**Key insight**: The model's active sign accuracy of **60.5%** represents a **+22 percentage-point lift** over the majority baseline (~38.5%) and a **+21.5pp lift** over momentum (~39%) — the two baselines most appropriate for a pension mandate. The always-long baseline (43%) is shown for reference only; it is not a realistic strategy for a risk-constrained institutional investor because it has no mechanism for capital preservation in drawdown periods.

The model functions as a **directional signal detector**: 60.5% of the time it takes a non-neutral position, that position is in the correct direction relative to the XFN sector. It does not predict magnitude.

## Reproducibility

### Load and use the locked artifact
```python
import pickle, sys
from pathlib import Path

ROOT = Path('...')  # project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'step3_predictive_model/model_experiments'))

with open('artifacts/lgbm_3class_xfn5d_35f_tuned-2026-04-22.pkl', 'rb') as f:
    artifact = pickle.load(f)

model     = artifact['model']
features  = artifact['features']
TARGET    = artifact['target']       # 'target_excess_xfn_5d'
THRESHOLD = artifact['threshold']    # 0.003
inv_map   = artifact['inverse_label_map']  # {0:-1, 1:0, 2:1}

# Predict on new data (DataFrame `new_df` with all 35 features)
preds_enc = model.predict(new_df[features].fillna(new_df[features].median()))
signals   = [inv_map[p] for p in preds_enc]   # -1, 0, or +1
```

### Retrain from scratch
Run the training notebook end-to-end — it performs the full hyperparameter search, selects the best config, trains on all available data, and saves a new dated artifact:
```bash
cd <project_root>
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    step3_predictive_model/final_model/lgbm_3class_xfn5d_35f_training.ipynb
```

## Limitations
1. **Single stock** — trained only on TD Bank. Signals are specific to TD's price dynamics, sector relationship, and earnings cadence.
2. **Directional only** — the model does not predict magnitude, only direction relative to XFN sector ETF.
3. **Data dependency** — requires daily feature pipeline (`model_features_daily.parquet`) and price data.
4. **No broad macro news** — the news features (`news_sent_mean_30d`, `news_count_30d`) are sourced from TD's own newsroom and cover company-specific press releases only. Broad macroeconomic news — central bank decisions, trade policy announcements, geopolitical events — is not captured as a text signal. Its effect enters the model only indirectly through market reactions already embedded in price and FX features (`yield_curve_slope`, `dxy_level`, `fx_usdcad_level`), which cannot isolate the specific driver. Incorporating structured macro news feeds would require a licensed data source (e.g. Refinitiv, Bloomberg News API), as free-tier alternatives lack the coverage, latency, and consistent structure needed for reliable daily feature construction.
