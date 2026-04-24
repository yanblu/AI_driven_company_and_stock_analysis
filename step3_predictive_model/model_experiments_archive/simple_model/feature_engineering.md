# Simple Model — Feature Engineering

**Model**: `lgbm_3class_xfn5d_simple`  
**Target**: 5-day excess return of TD vs XFN sector ETF (`target_excess_xfn_5d`)  
**Feature count**: 15  
**Design principle**: Only raw-data-driven, directly explainable features. No technical indicators (no R², no correlations, no beta, no VWMA). No macro/global-risk proxies. All features must have a clear, sector-relative rationale.

---

## Feature Groups

### Group 1 — Price & Relative Momentum (4 features)

| Feature | Construction | Rationale |
|---|---|---|
| `td_return_5d` | 5-day log return of TD | Short-term price momentum; raw and directly observable |
| `td_return_20d` | 20-day log return of TD | Medium-term momentum; captures recent trend |
| `td_vs_xfn_5d` | TD 5-day return − XFN 5-day return | Recent sector-relative momentum; directly measures the target concept over a trailing window |
| `td_dist_52w_high` | (52-week high − current price) / 52-week high | Mean-reversion anchor; pure price level signal, no statistical fitting required |

**Dropped from 35f**: `td_volatility_20d` (sector-wide effect, no idiosyncratic signal), `td_vs_tsx_5d` (redundant with `td_vs_xfn_5d` for XFN-relative target), all 5 TA features (`td_rsqr_60d`, `td_wvma_5d`, `td_wvma_20d`, `td_corr_pv_5d`, `td_beta_20d`) — these require statistical fitting that makes them hard to explain and less robust at short horizons.

---

### Group 2 — News (3 features)

| Feature | Construction | Rationale |
|---|---|---|
| `news_sent_mean_30d` | Rolling 30-day average news sentiment score | Recent public sentiment signal; stable enough without further decay |
| `news_count_30d` | Rolling 30-day article count | News intensity; high count may indicate unusual attention |
| `days_since_last_news` | Calendar days since the last news article | Recency control; avoids stale sentiment being treated as fresh |

**Dropped from 35f**: `evt_news_tone` and `evt_news_flow` — these are news-decay-weighted variants already captured here. Keeping raw versions avoids double-counting and improves explainability.

---

### Group 3 — Earnings Call Timing (2 features)

| Feature | Construction | Rationale |
|---|---|---|
| `days_since_call` | Calendar days since last earnings call | Controls for information staleness across all NLP features |
| `is_earnings_week` | Binary: 1 if the current week contains an earnings call | Identifies elevated uncertainty / volatility around announcements |

---

### Group 4 — NLP Event Signals (6 features)

All NLP features are forward-filled from the most recent earnings call and multiplied by `call_decay = exp(−days_since_call / 20)` (half-life ≈ 14 days), consistent with the final model approach.

| Feature | Construction | Raw inputs |
|---|---|---|
| `evt_exec_tone` | `mean(ceo_prep_sentiment, cfo_prep_sentiment) × call_decay` | `transcript_ceo_prep_sentiment_mean_ffill`, `transcript_cfo_prep_sentiment_mean_ffill` |
| `evt_framing_gap` | `framing_gap_ffill × call_decay` | `framing_gap_ffill` |
| `evt_guidance_topic` | `topic_guidance_share_ffill × call_decay` | `topic_guidance_share_ffill` |
| `evt_guidance_sentiment` | `topic_guidance_sentiment_ffill × call_decay` | `topic_guidance_sentiment_ffill` |
| `evt_aml_topic` | `topic_regulatory_AML_share_ffill × call_decay` | `topic_regulatory_AML_share_ffill` |
| `evt_aml_sentiment` | `topic_regulatory_AML_sentiment_ffill × call_decay` | `topic_regulatory_AML_sentiment_ffill` |

**Design decisions vs. 35f**:
- `evt_exec_tone` replaces separate `evt_ceo_tone` + `evt_cfo_tone`: CEO and CFO tone are highly correlated; averaging reduces noise and halves feature count.
- `evt_guidance_topic` + `evt_guidance_sentiment` replace `evt_guidance_strength` (product of share × sentiment) and `evt_guidance_shift` (quarter-over-quarter delta). The simple versions let the model learn interactions from raw components rather than pre-engineering them. The loss of the explicit shift signal is the main performance trade-off.
- `evt_aml_topic` + `evt_aml_sentiment` replace `evt_aml_pressure` (share × negative sentiment product) and `evt_aml_shift` (absolute quarter-over-quarter delta). Same rationale as guidance.
- `evt_framing_gap` is kept as-is — it is a single, interpretable gap between management tone and filed report tone, already simple.

**Dropped from 35f**: `evt_macro_topic`, `evt_topic_entropy`. These had low SHAP importance and are sector-wide signals that would affect all XFN banks similarly, adding little idiosyncratic value.

---

## Features Dropped vs. 35f — Summary

| Feature | Group | Reason for dropping |
|---|---|---|
| `td_volatility_20d` | Price | Sector-wide effect; no idiosyncratic signal |
| `td_vs_tsx_5d` | Price | Redundant with `td_vs_xfn_5d` for XFN-relative target |
| `td_corr_pv_20d` | TA | Technical; hard to explain; was #2 SHAP in 35f |
| `td_volume_change_20d` | Volume | Volume change has limited predictive power for 5-day price-level sector-relative target |
| `td_rsqr_60d` | TA | Technical; hard to explain; was #1 SHAP in 35f |
| `td_wvma_5d` | TA | Technical |
| `td_wvma_20d` | TA | Technical |
| `td_corr_pv_5d` | TA | Technical |
| `td_beta_20d` | TA | Technical; sector-wide effect |
| `yield_curve_slope` | Macro | Affects all banks similarly; already reflected in XFN benchmark |
| `yield_10y_level` | Macro | Redundant given yield curve slope; sector-wide |
| `vix_volatility_20d` | Macro | Sector-wide risk-off signal; no idiosyncratic value |
| `dxy_level` | Macro | Sector-wide; currency effect visible in XFN |
| `gold_level` | Macro | Not requested; sector-wide |
| `fx_usdcad_level` | Macro | Sector-wide; absorbed by XFN benchmark |
| `evt_ceo_tone` | NLP | Replaced by combined `evt_exec_tone` |
| `evt_cfo_tone` | NLP | Replaced by combined `evt_exec_tone` |
| `evt_aml_pressure` | NLP | Replaced by `evt_aml_topic` + `evt_aml_sentiment` |
| `evt_aml_shift` | NLP | Replaced by `evt_aml_topic` + `evt_aml_sentiment` |
| `evt_guidance_strength` | NLP | Replaced by `evt_guidance_topic` + `evt_guidance_sentiment` |
| `evt_guidance_shift` | NLP | Replaced by `evt_guidance_topic` + `evt_guidance_sentiment` |
| `evt_macro_topic` | NLP | Low SHAP importance; sector-wide signal |
| `evt_topic_entropy` | NLP | Low SHAP importance; not idiosyncratic |
| `evt_news_tone` | NLP | Replaced by raw `news_sent_mean_30d` (news-decay double-counts) |
| `evt_news_flow` | NLP | Replaced by raw `news_count_30d` (news-decay double-counts) |

---

## Known Performance Trade-offs

1. **Loss of #1 and #2 SHAP features** (`td_rsqr_60d`, `td_corr_pv_20d`): These were the most predictive features in the 35f model but are dropped for explainability. Expect a material reduction in ASA.
2. **No explicit guidance shift signal**: `evt_guidance_shift` was among the top NLP features. Replacing it with raw components (`evt_guidance_topic`, `evt_guidance_sentiment`) reduces the explicit change signal — the model must infer it.
3. **No macro risk context**: Removing yield curve, VIX, DXY, and FX reduces the model's awareness of regime-level risk. While these add noise for a sector-relative target, they can still provide useful context during extreme macro events.
