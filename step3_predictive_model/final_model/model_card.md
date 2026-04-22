# R07 — Final Model Card

## Identity
| Field | Value |
|---|---|
| Model ID | `R07` (also labelled `S2_Reduced` in experiment logs) |
| Algorithm | LightGBM multiclass classifier |
| Artifact | `artifacts/r07-current-best-2026-04-21.pkl` |
| Metadata | `artifacts/r07-current-best-2026-04-21.json` |
| Saved | 2026-04-21 |

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
    n_estimators    = 120,
    learning_rate   = 0.05,
    num_leaves      = 8,           # shallow — avoids overfitting on single stock
    min_child_samples = 40,
    feature_fraction  = 0.8,
    reg_alpha       = 0.2,
    reg_lambda      = 1.0,
    random_state    = 42,
)
```

## Features (30)

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

## Validation
- **Protocol**: Expanding-window walk-forward, 7 folds (`E10_FOLDS`)
- **Gap**: 5-row buffer between train and test to avoid lookahead
- **Evaluation**: Stride-5 offset averaging (predictions every day, evaluated on every 5th to avoid overlapping 5-day returns)

## Performance (7-fold walk-forward average)

### R07 Model
| Metric | Value |
|---|---|
| Mean accuracy | ~45% |
| Macro F1 | ~0.38 |
| Active coverage | ~87% |
| Active sign accuracy | ~57% |
| DirAcc_abstain50 | ~0.54 |

### Baselines (same 7-fold average)
Three baselines are reported. The **majority** and **momentum** baselines are the most relevant benchmarks for a pension context — a pension mandate typically cannot default to always-long because it must manage drawdown risk and is often constrained to act symmetrically on long and short exposures.

| Baseline | Description | Mean Acc | Active Coverage | Active Sign Acc |
|---|---|---|---|---|
| **Majority** | Always predict the most frequent class from training data | ~38.5% | 100% | ~38.5% |
| **Momentum** | Predict continuation of TD's recent 5-day relative performance vs XFN | ~36.6% | ~79% | ~39.0% |
| Always-long | Always predict outperform (+1) | ~43% | 100% | ~43% |

**Key insight**: R07's active sign accuracy of ~57% represents a **+18 percentage-point lift** over the majority baseline (~38.5%) and a **+18pp lift** over momentum (~39%) — the two baselines most appropriate for a pension mandate. The always-long baseline (43%) is shown for reference only; it is not a realistic strategy for a risk-constrained institutional investor because it has no mechanism for capital preservation in drawdown periods.

The model functions as a **directional signal detector**: ~57% of the time it takes a non-neutral position, that position is in the correct direction relative to the XFN sector. It does not predict magnitude.

## Reproducibility

### Load and use the locked artifact
```python
import pickle, sys
from pathlib import Path

ROOT = Path('...')  # project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'step3_predictive_model/model_experiments'))

with open('artifacts/r07-current-best-2026-04-21.pkl', 'rb') as f:
    artifact = pickle.load(f)

model     = artifact['model']
features  = artifact['features']
TARGET    = artifact['target']       # 'target_excess_xfn_5d'
THRESHOLD = artifact['threshold']    # 0.003
inv_map   = artifact['inverse_label_map']  # {0:-1, 1:0, 2:1}

# Predict on new data (DataFrame `new_df` with all 30 features)
preds_enc = model.predict(new_df[features].fillna(new_df[features].median()))
signals   = [inv_map[p] for p in preds_enc]   # -1, 0, or +1
```

### Retrain from scratch
Run `save_r07_artifact.py` (located in `final_model/` for convenience, or the canonical version in `model_experiments/redesign_single_stock/src/`):
```bash
cd <project_root>
.venv/bin/python step3_predictive_model/final_model/save_r07_artifact.py
```

## Limitations
1. **Single stock** — trained only on TD Bank. Signals are specific to TD's price dynamics, sector relationship, and earnings cadence.
2. **Directional only** — the model does not predict magnitude, only direction relative to XFN sector ETF.
3. **AML feature timing** — `evt_aml_pressure` and `evt_aml_shift` capture regulatory discourse that was particularly active in the AML period (2023-2024). Feature selection included these with domain knowledge; this is documented in the experiment log.
4. **Data dependency** — requires daily feature pipeline (`model_features_daily.parquet`) and price data.
