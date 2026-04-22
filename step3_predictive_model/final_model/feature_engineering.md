# lgbm_3class_xfn5d_30f — Feature Engineering

This document describes how each of the 30 features used in the final model is constructed. All features are computed at **daily grain** and stored in `model_features_daily.parquet`.

---

## Decay function (used by all NLP event features)

All 11 NLP event features share an exponential decay anchored to the last earnings call:

```
call_decay = exp(−days_since_call / 20)
```

The half-life of 20 trading days means a signal that was 1.0 on call day has decayed to ~0.5 after 4 calendar weeks. This reflects how earnings-call information enters and fades from the market. A parallel decay with a 15-day half-life is used for news-anchored features:

```
news_decay = exp(−days_since_last_news / 15)
```

---

## Block 1 — Market / Price (8 features)

These features describe TD's own price behaviour and its relative position in the market. All are computed from daily adjusted close prices.

| Feature | Construction | Rationale |
|---|---|---|
| `td_return_5d` | 5-day log return of TD.TO | Recent price momentum; captures short-term trend |
| `td_return_20d` | 20-day log return of TD.TO | Medium-term trend; offsets noise in the 5d signal |
| `td_volatility_20d` | Rolling 20-day annualised realised volatility | Regime indicator; high vol compresses signal-to-noise |
| `td_vs_xfn_5d` | TD 5-day return minus XFN 5-day return | Most direct measure of recent sector-relative momentum; also used as the momentum baseline |
| `td_vs_tsx_5d` | TD 5-day return minus TSX 5-day return | Broad-market-relative momentum; separates sector from market effect |
| `td_corr_pv_20d` | 20-day rolling Pearson correlation between TD price and volume | High correlation often signals accumulation or distribution; sign and magnitude both informative |
| `td_volume_change_20d` | 20-day z-score of daily volume change | Unusual volume often precedes price moves; normalised to remove secular trends |
| `td_dist_52w_high` | `(TD adj_close − 52w high) / 52w high` | Convex mean-reversion signal; far from highs can indicate either distress or value |

---

## Block 2 — Rates / Macro (2 features)

These features summarise the Canadian interest rate environment, which has outsized influence on bank valuations.

| Feature | Construction | Rationale |
|---|---|---|
| `yield_curve_slope` | 10-year Government of Canada yield minus 2-year yield | Steeper curve is generally supportive for bank net interest margins |
| `yield_10y_level` | 10-year Government of Canada yield (level) | Captures the absolute rate environment, which drives asset repricing and funding cost |

---

## Block 3 — Global Risk (4 features)

These features capture cross-asset risk sentiment that affects financials as a sector.

| Feature | Construction | Rationale |
|---|---|---|
| `vix_volatility_20d` | 20-day realised volatility of the VIX index itself | A "volatility of volatility" measure; elevated values often precede risk-off periods |
| `dxy_level` | US Dollar Index level | USD strength affects TD's US operations and cross-border capital flows |
| `gold_level` | Gold spot price level | Safe-haven proxy; rising gold often coincides with bank sector weakness |
| `fx_usdcad_level` | USD/CAD spot rate level | Direct FX risk exposure for TD's US segment and Canadian investor perspective |

---

## Block 4 — News Flow (3 features)

These features track the raw volume and tone of news coverage of TD, independent of earnings calls.

| Feature | Construction | Rationale |
|---|---|---|
| `news_sent_mean_30d` | Rolling 30-day mean of LLM-scored news sentiment (raw, without decay) | Sustained news tone reflects the information environment that analysts and traders are reading |
| `news_count_30d` | Rolling 30-day count of TD news articles | News burst is an independent signal; volume spikes often coincide with significant events |
| `days_since_last_news` | Calendar days elapsed since the most recent news article | Used as the anchor for `news_decay`; also a standalone feature — a long news silence can itself be informative |

---

## Block 5 — Timing (2 features)

These features encode the calendar position of the observation relative to earnings events.

| Feature | Construction | Rationale |
|---|---|---|
| `days_since_call` | Trading days since the most recent earnings call | Primary decay anchor for all NLP event features; also directly informative — the model can learn that signals lose predictive power as the call recedes |
| `is_earnings_week` | Binary flag: 1 if within 5 trading days of an earnings release | Captures the asymmetric information environment around reporting dates |

---

## Block 6 — NLP Event Features (11 features)

These 11 features are the core NLP contribution of the project. They are constructed from LLM-scored earnings call transcripts (GPT-4, scored at quarterly grain) and projected to daily grain using forward-fill followed by exponential decay.

**Design principle**: rather than feeding the model a wide block of ~99 forward-filled quarterly NLP columns (the prior approach), each feature compresses one analytical question into a single decayed scalar. This matches how earnings-call information actually enters the market — strongest right after the call, then fading — and avoids the high correlation and redundancy of the raw NLP block.

### `evt_ceo_tone`
```
transcript_ceo_prep_sentiment_mean_ffill × call_decay
```
CEO prepared-statement sentiment from the latest call, decayed by time since the call. Captures the overall tone management wants to project.

---

### `evt_cfo_tone`
```
transcript_cfo_prep_sentiment_mean_ffill × call_decay
```
Same construction for the CFO. CFO sentiment often diverges from CEO sentiment at inflection points and was found in Step 2 analysis to lead directional changes in analyst tone.

---

### `evt_framing_gap`
```
framing_gap_ffill × call_decay
```
The `framing_gap` raw column is the difference between CEO prepared-statement sentiment and the sentiment of the filed investor relations report for the same quarter. A positive gap (management sounds more optimistic than the documents) has historically been a cautionary signal.

---

### `evt_aml_pressure`
```
topic_regulatory_AML_share_ffill × (1 − topic_regulatory_AML_sentiment_ffill) × call_decay
```
Combines two signals: how much of the call was about AML/regulatory topics, and how negative that discussion was. The `(1 − sentiment)` term makes the product large when AML prominence is high **and** sentiment is low — i.e., genuine regulatory stress, not a routine legal update.

---

### `evt_aml_shift`
```
(|topic_regulatory_AML_share_delta| + |topic_regulatory_AML_sentiment_delta|) × call_decay
```
Captures quarter-over-quarter *change* in the AML narrative, regardless of direction. A sudden jump in AML share or tone (positive or negative) signals a new development and is informative independent of the level.

---

### `evt_guidance_strength`
```
topic_guidance_share_ffill × topic_guidance_sentiment_ffill × call_decay
```
Forward-looking guidance discussion weighted by how positive it was. A call with a large, optimistic guidance section scores high; a call with minimal or cautious guidance scores low.

---

### `evt_guidance_shift`
```
(topic_guidance_share_delta + topic_guidance_sentiment_delta) × call_decay
```
Quarter-over-quarter change in guidance emphasis and tone (both signed, not absolute). A positive shift means management is devoting more time to forward guidance and/or is more optimistic about it.

---

### `evt_macro_topic`
```
topic_macro_outlook_share_ffill × (topic_macro_outlook_sentiment_ffill − 0.5) × call_decay
```
How much management discussed the macro outlook, weighted by whether that discussion was above-neutral sentiment. The `−0.5` centring means neutral sentiment (0.5) contributes zero; only distinctly positive or negative macro commentary moves the feature.

---

### `evt_topic_entropy`
```
topic_entropy_ffill × call_decay
```
Shannon entropy of the topic distribution in the latest call transcript. Low entropy means the call was dominated by a few topics (e.g., AML); high entropy means a broad, balanced discussion. Entropy changes around significant events.

---

### `evt_news_tone`
```
news_sent_mean_30d × news_decay
```
The 30-day rolling news sentiment weighted by recency of the latest article. Unlike `news_sent_mean_30d` (which is in Block 4), this version discounts the signal when no fresh articles have appeared, reflecting that stale sentiment is less actionable.

---

### `evt_news_flow`
```
log(1 + news_count_30d) × news_decay
```
Log-scaled 30-day news volume, decayed by news recency. Log scaling prevents a brief burst of articles from dominating; the recency decay ensures the feature falls off when the burst has passed.

---

## Summary table

| # | Feature | Block | Raw source(s) | Transformation |
|---|---|---|---|---|
| 1 | `td_return_5d` | Market | TD adj_close | 5-day log return |
| 2 | `td_return_20d` | Market | TD adj_close | 20-day log return |
| 3 | `td_volatility_20d` | Market | TD adj_close | Rolling 20-day annualised vol |
| 4 | `td_vs_xfn_5d` | Market | TD, XFN adj_close | TD 5d return − XFN 5d return |
| 5 | `td_vs_tsx_5d` | Market | TD, TSX adj_close | TD 5d return − TSX 5d return |
| 6 | `td_corr_pv_20d` | Market | TD price, volume | 20-day price-volume correlation |
| 7 | `td_volume_change_20d` | Market | TD volume | 20-day z-score of volume change |
| 8 | `td_dist_52w_high` | Market | TD adj_close | (price − 52w high) / 52w high |
| 9 | `yield_curve_slope` | Rates / Macro | GoC 10Y, 2Y | 10Y − 2Y yield |
| 10 | `yield_10y_level` | Rates / Macro | GoC 10Y | Level |
| 11 | `vix_volatility_20d` | Global Risk | VIX | Rolling 20-day vol of VIX |
| 12 | `dxy_level` | Global Risk | DXY | Level |
| 13 | `gold_level` | Global Risk | Gold spot | Level |
| 14 | `fx_usdcad_level` | Global Risk | USD/CAD | Level |
| 15 | `news_sent_mean_30d` | News Flow | LLM news scores | 30-day rolling mean |
| 16 | `news_count_30d` | News Flow | Article timestamps | 30-day rolling count |
| 17 | `days_since_last_news` | News Flow | Article timestamps | Days since latest article |
| 18 | `days_since_call` | Timing | Earnings calendar | Days since last earnings call |
| 19 | `is_earnings_week` | Timing | Earnings calendar | Binary flag, ±5 days of release |
| 20 | `evt_ceo_tone` | NLP Event | CEO prep sentiment (LLM) | Sentiment × call_decay |
| 21 | `evt_cfo_tone` | NLP Event | CFO prep sentiment (LLM) | Sentiment × call_decay |
| 22 | `evt_framing_gap` | NLP Event | framing_gap (CEO vs IR doc) | Gap × call_decay |
| 23 | `evt_aml_pressure` | NLP Event | AML topic share + sentiment | share × (1 − sentiment) × call_decay |
| 24 | `evt_aml_shift` | NLP Event | AML share delta, sentiment delta | (\|Δshare\| + \|Δsentiment\|) × call_decay |
| 25 | `evt_guidance_strength` | NLP Event | Guidance share + sentiment | share × sentiment × call_decay |
| 26 | `evt_guidance_shift` | NLP Event | Guidance share delta, sentiment delta | (Δshare + Δsentiment) × call_decay |
| 27 | `evt_macro_topic` | NLP Event | Macro topic share + sentiment | share × (sentiment − 0.5) × call_decay |
| 28 | `evt_topic_entropy` | NLP Event | Topic distribution entropy | Entropy × call_decay |
| 29 | `evt_news_tone` | NLP Event | news_sent_mean_30d | Sentiment × news_decay |
| 30 | `evt_news_flow` | NLP Event | news_count_30d | log(1 + count) × news_decay |
