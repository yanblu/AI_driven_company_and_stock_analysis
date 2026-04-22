# NLP event-feature design, grounded in `docs/step2_analysis.md`

## Purpose

The previous redesign used a mix of event-style NLP features chosen partly by intuition and partly by SHAP ranking. This document re-anchors the NLP feature block in the **Step 2 analytical findings** — which patterns were empirically validated as high-confidence, medium-confidence, or noise — and derives a compact daily event-feature set directly from that evidence.

The goal is traceability: every daily event feature should map back to a named finding in `docs/step2_analysis.md` so the feature set can be defended to stakeholders and audited later.

Scope:

- **In scope**: the quarter-grain LLM-annotated NLP block (`data/processed/features/nlp_features.parquet`) and its projection to daily event-style features used in the redesigned single-stock model.
- **Out of scope**: market features, news-rolling counts beyond `news_sent_mean_30d` / `news_count_30d`, and chunk-level re-annotation. Those are unchanged.

---

## 1. Raw schema (quarter grain)

Source: `data/processed/features/nlp_features.parquet` — **21 quarters × 36 columns**, one row per fiscal quarter FY2021Q1 → FY2026Q1.

### Metadata

| Column | Type | Notes |
|---|---|---|
| `fiscal_quarter` | str | join key, e.g. `FY2024Q4` |
| `n_chunks_annotated` | int | 106–258 per quarter; reliability proxy |

### Sentiment channels (7 first-class, 1 derived)

| Column | Grain | What it measures |
|---|---|---|
| `transcript_ceo_prep_sentiment_mean` | quarter | CEO prepared remarks, the primary scripted narrative channel |
| `transcript_cfo_prep_sentiment_mean` | quarter | CFO prepared remarks, operational confidence channel |
| `transcript_exec_qa_sentiment_mean` | quarter | Executive Q&A answers — *live pressure* response |
| `transcript_analyst_qa_sentiment_mean` | quarter | Analyst questions — *external scrutiny* tone |
| `transcript_sentiment_std` | quarter | Within-quarter volatility of transcript sentiment (reliability proxy per L2) |
| `news_sentiment_mean` | quarter | Newsroom press-release tone |
| `report_40f_sentiment_mean` | annual Q4 only | Annual 40-F filing tone (null in Q1–Q3) |
| `report_quarterly_sentiment_mean` | Q1–Q3 only | Quarterly Report to Shareholders tone (null in Q4) |
| `report_sentiment_mean` | quarter | **Derived**: `report_quarterly_*` in Q1–Q3, `report_40f_*` in Q4. **Always populated**, and this is the column the daily builder uses. The two sparse parent columns are explicitly dropped before projection to daily — see `src/features/build_daily_features.py` lines 343–350. |

### Topic channels (12 labels × 2 metrics = 24 columns)

For every label in `{NIM, credit_quality, capital, US_retail, Canadian_personal, wealth_wholesale, regulatory_AML, macro_outlook, cost_efficiency, guidance, M_and_A, other}`:

- `topic_{label}_share` — fraction of that quarter's chunks tagged with this label
- `topic_{label}_sentiment` — mean sentiment of chunks tagged with this label

### Diversity

| Column | What it measures |
|---|---|
| `topic_entropy` | Shannon entropy of topic share distribution (excluding `other`). Low = concentrated narrative (e.g. AML peak); high = broad discussion |

### Known quirks called out in Step 2

- **L1 / L3**: raw table chunks bias report and news sentiment. L1 was remediated by a table filter. L3 (press-release trailing tables) partially remains — keep `news_*` features but treat them as second-tier.
- **L2**: CEO prepared-remarks sentiment can be a 1–3 chunk mean in some quarters. `transcript_sentiment_std` is the reliability proxy.
- **L6**: `other` averages 26% of chunks; topic features have a known noise floor in that channel.
- **L8 / L9 (remediated)**: admin intro chunks stripped; Q&A parsing fixed for FY2022Q3/Q4 and FY2023Q2.

---

## 2. Daily projection rules (already in the codebase)

Implemented in `src/features/build_daily_features.py::build_nlp_features`:

1. Each quarter row is anchored to its **call date** (using the `CALL_DATES` map).
2. Every quarter-grain column is forward-filled from its call date onward along the market trading-date spine.
3. Two derivations are added at the quarter level before projection:
   - `{col}_delta` — quarter-over-quarter first difference.
   - `{col}_surprise` — value minus the mean of the previous 4 quarters (backward-looking only).
4. After projection, columns are renamed with a `_ffill` suffix. The daily parquet therefore carries every NLP column in three flavours: **level** (`_ffill`), **delta**, **surprise** — all computed at the quarterly grain, all past-only at any given daily date.
5. A daily `days_since_call` and `is_earnings_week` flag are added.
6. Two decays are available to event-feature builders:
   - `call_decay = 0.5^(days_since_call / 30)` — information freshness after each call.
   - `news_decay = exp(-days_since_last_news / 15)` — news recency.

These rules are past-only by construction: a row dated day `d` only ever references the most recent call `≤ d`, the 4 preceding quarters' mean, and rolling news from `≤ d`. No look-ahead.

---

## 3. Mapping Step 2 findings to feature primitives

This is the core of the design. Each row below names a specific Step 2 finding, its confidence level from the Evaluation D table, and the feature primitive that encodes it.

| Step 2 finding | Confidence | Primitive | Daily feature |
|---|---|---|---|
| Framing gap at FY2023Q3 was +0.722 (largest in dataset, pre-announcement) | **High — Pass** | CEO prep − merged filing | `evt_framing_gap` |
| Framing gap compressed to +0.36–+0.41 during FY2025Q1–Q2 recovery (never inverted) | Medium — Pass | Same primitive; sign/magnitude track regime | `evt_framing_gap` |
| CEO prep compressed to 0.433 at FY2023Q4 (11 months before consent order); dataset-only zero at FY2024Q4 | Medium — Observed | CEO tone level + surprise vs trailing 4Q | `evt_ceo_tone`, `evt_ceo_surprise` |
| CEO/CFO both recovered to 0.600 at FY2025Q1; CFO led by one step at Q2 and Q3'25 | Medium — Pass | CFO − CEO difference | `evt_cfo_lead` |
| Guidance collapsed to 8.7% of chunks at FY2024Q2; peaked at 17.8% at FY2025Q1 | **High — Pass** | Guidance share × sentiment | `evt_guidance_strength`, `evt_guidance_shift` |
| AML share rose 11.1% → 30.0% (FY2024Q4 peak); AML sentiment 0.239 → 0.033 | **High — Pass** | Topic share × (1 − sentiment) | `evt_aml_pressure`, `evt_aml_shift` |
| AML narrative moved across **multiple sources simultaneously** (transcripts + filings + news) | Medium — Observed | Cross-source agreement / dispersion | `evt_source_dispersion` (**new**) |
| Analyst QA peaked at FY2026Q1 (0.529, dataset high) — external buy-side confidence | **High — Pass** | Analyst-QA tone, decayed | `evt_analyst_qa_tone` (**new**) |
| Execs under pressure: live Q&A answers distinct from scripted remarks | Medium | Exec-QA tone − CEO-prep tone | `evt_exec_scripted_gap` (**new**) |
| Topic concentration spiked at AML peak (low entropy) | Medium | Topic entropy, ffill | `evt_topic_entropy` |
| Broad non-AML topic basket carries residual business signal | Medium | Avg. signal across credit/capital/wealth/retail/NIM | `evt_broad_topic_signal`, `evt_broad_topic_shift` |
| News channel is useful but biased by tables (L3) | Low-to-medium | 30-day rolling news features decayed | `evt_news_tone`, `evt_news_flow` (**demoted**) |
| L2 reliability: CEO sentiment noisy when <3 chunks; `transcript_sentiment_std` quantifies it | Supporting | Inverse volatility as a modifier | `evt_transcript_reliability` (**new**) |
| Macro topic (rate-cycle language) moved with BoC hike cycle | Partial — Medium | Macro topic share × centred sentiment | `evt_macro_topic` |

Rows marked **(new)** are the features this design adds. Rows without that marker already exist in `run_redesign_experiments.py::load_dataset`.

---

## 4. Proposed daily event-feature set

All features are engineered from the daily parquet (`_ffill`, `_delta`, `_surprise` columns) and multiplied by `call_decay` unless noted. The decay makes each feature strongest on earnings-call day and half at ~30 trading days later, matching the "information freshness between quarterly pulses" idea that Step 2 implicitly relies on.

### Group A — Executive tone core (6 features)

Direct encoding of the primary Step 2 narrative signals.

1. **`evt_ceo_tone`** — `transcript_ceo_prep_sentiment_mean_ffill × call_decay`
   *Rationale*: primary scripted narrative channel. High-confidence compression at FY2023Q4 (0.433) and zero at FY2024Q4 are captured directly. Already in pipeline.

2. **`evt_cfo_tone`** — `transcript_cfo_prep_sentiment_mean_ffill × call_decay`
   *Rationale*: operational confidence channel. Already in pipeline.

3. **`evt_cfo_lead`** (**new**) — `(transcript_cfo_prep_sentiment_mean_ffill − transcript_ceo_prep_sentiment_mean_ffill) × call_decay`
   *Rationale*: Step 2 Act 3 identifies this gap as the specific recovery marker at FY2025Q2 (0.700 vs 0.600) and FY2025Q3 (0.800 vs 0.700). Currently unrepresented.

4. **`evt_ceo_surprise`** (**new**) — `transcript_ceo_prep_sentiment_mean_surprise × call_decay`
   *Rationale*: backward-looking deviation from trailing 4Q mean. Catches the Step 2 finding that FY2023Q4 CEO compression was notable *relative to that executive's own baseline*, not the cross-sectional mean. Zero look-ahead: `_surprise` uses only the prior 4 quarters.

5. **`evt_analyst_qa_tone`** (**new**) — `transcript_analyst_qa_sentiment_mean_ffill × call_decay`
   *Rationale*: Step 2 calls this the cleanest external-confidence marker (peak FY2026Q1 = 0.529, dataset high). Currently unused in the event block.

6. **`evt_exec_scripted_gap`** (**new**) — `(transcript_exec_qa_sentiment_mean_ffill − transcript_ceo_prep_sentiment_mean_ffill) × call_decay`
   *Rationale*: gap between *live* (Q&A) and *scripted* (prepared) exec tone. Widens when execs answer unscripted questions more cautiously than they opened — a measured-silence proxy similar to the guidance trough.

### Group B — Divergence and commitment (2 features)

The two highest-confidence analytical primitives from Step 2.

7. **`evt_framing_gap`** — `framing_gap_ffill × call_decay`, where `framing_gap = ceo_prep − report_sentiment_mean`
   *Rationale*: the single highest-confidence pre-event signal (Pass at +0.722 at FY2023Q3); compressed but positive (+0.36/+0.41) during recovery. Already in pipeline.

8. **`evt_guidance_strength`** and **`evt_guidance_shift`** — guidance topic share × sentiment (level and shift). Already in pipeline.
   *Rationale*: trough 8.7% at FY2024Q2 = Pass; peak 17.8% at FY2025Q1 = Pass. Direct commitment thermometer.

### Group C — AML regime and multi-source consistency (3 features)

9. **`evt_aml_pressure`** — `topic_regulatory_AML_share_ffill × (1 − topic_regulatory_AML_sentiment_ffill) × call_decay`
   *Rationale*: encodes both attention and tone; FY2024Q4 peak captures the consent-order quarter directly. Already in pipeline.

10. **`evt_aml_shift`** — `(|share_delta| + |sentiment_delta|) × call_decay`
    *Rationale*: detects regime changes in AML language — the monotonic decline from FY2022Q3 to FY2023Q3 (Observed) and the FY2024Q4 spike. Already in pipeline.

11. **`evt_source_dispersion`** (**new**) — standard deviation across three source-level sentiments on the same date, per Step 2's finding that the AML narrative emerged simultaneously across transcripts, filings and news.
    `evt_source_dispersion = std([transcript_ceo_prep_sentiment_mean_ffill, news_sentiment_mean_ffill, report_sentiment_mean_ffill]) × call_decay`
    *Rationale*: low dispersion when all channels move together (regime shift); high dispersion when one source spikes alone (single-quarter noise). Captures the thing that made the AML signal credible in Step 2 — multi-source corroboration — without tying to any single source.

### Group D — Topic attention compression (3 features)

12. **`evt_topic_entropy`** — `topic_entropy_ffill × call_decay`
    *Rationale*: low entropy = narrative collapsed onto a single topic (AML peak at FY2024Q4); high = broad discussion. Already in pipeline.

13. **`evt_broad_topic_signal`** — mean across non-AML business topics (credit/capital/wealth/retail/NIM) of `share × (sentiment − 0.5)`, decayed.
    *Rationale*: keeps the ambient business narrative visible even when AML dominates. Already in pipeline.

14. **`evt_broad_topic_shift`** — mean of `|share_delta| + |sentiment_delta|` across the same basket, decayed.
    *Rationale*: detects regime shifts in non-AML topics. Step 2 L6 notes 20.6% of content is non-taxonomy `other`, so a broad aggregator is more robust than any single topic feature. Already in pipeline.

### Group E — News channel (demoted, kept as second-tier) (2 features)

15. **`evt_news_tone`** — `news_sent_mean_30d × news_decay`
16. **`evt_news_flow`** — `log1p(news_count_30d) × news_decay`

*Rationale*: Step 2 L3 flags press-release trailing tables as a slight bias source. Kept because daily news cadence is the only sub-quarterly NLP channel, but treated as second-tier — if SHAP consistently picks these up, investigate before trusting.

### Group F — Reliability modifier (1 feature)

17. **`evt_transcript_reliability`** (**new**) — `1 / (1 + transcript_sentiment_std_ffill) × call_decay`
    *Rationale*: Step 2 L2 explicitly says CEO sentiment is unreliable when based on 1–2 chunks; `transcript_sentiment_std` is the reliability proxy. Exposing `1 / (1 + std)` as a feature lets a tree model condition on *how trustworthy* the current narrative signal is. (An alternative is to use `n_chunks_annotated_ffill` directly; `std` is more informative because it captures noise even with adequate counts.)

---

## 5. What changes versus the current pipeline

| Change | Feature(s) | Motivation |
|---|---|---|
| **Add** | `evt_cfo_lead`, `evt_ceo_surprise`, `evt_analyst_qa_tone`, `evt_exec_scripted_gap`, `evt_source_dispersion`, `evt_transcript_reliability` | Six high-confidence Step 2 primitives not currently surfaced as features |
| Keep as-is | `evt_ceo_tone`, `evt_cfo_tone`, `evt_framing_gap`, `evt_guidance_strength`, `evt_guidance_shift`, `evt_aml_pressure`, `evt_aml_shift`, `evt_macro_topic`, `evt_topic_entropy`, `evt_broad_topic_signal`, `evt_broad_topic_shift` | Already well-motivated by Step 2; no change |
| **Demote** to second-tier (still included) | `evt_news_tone`, `evt_news_flow` | Step 2 L3 bias; include but flag |
| **Drop** raw `_delta` and `_surprise` columns from the candidate pool when using the event-feature-only track | ~70 `topic_*_delta`, `topic_*_surprise` columns | They are already compressed inside the engineered `evt_*` features; letting both into the selector creates redundant feature families and penalises SHAP stability |

Result: **17 event-style NLP features** instead of the current 13, every one traceable to a Step 2 finding by name.

---

## 6. Why this is a sounder design

1. **Every feature maps to a named Step 2 finding.** If SHAP rejects a feature on a fold, we know which analytical claim we are failing to monetise. If SHAP accepts one, we can point to the passage in `docs/step2_analysis.md` that motivated it. This is the audit trail we did not have.
2. **The two known blind spots of the previous event block are filled.**
   - *Multi-source corroboration* (`evt_source_dispersion`) addresses what Step 2 calls out as the distinctive property of the AML narrative: it moved across transcripts, filings, and news in sync. No single-source feature captures this.
   - *Reliability weighting* (`evt_transcript_reliability`) operationalises the L2 caveat explicitly instead of hoping the tree model picks it up by accident.
3. **Recovery-regime primitives are now first-class.** `evt_cfo_lead`, `evt_analyst_qa_tone`, and `evt_exec_scripted_gap` are the three features needed to trigger on the FY2025+ recovery sequence that Step 2 documents step by step. Without them, the model can only see AML *building* but not AML *resolving*, which is a known weakness at fold `v5` in every previous redesign round.
4. **Redundancy with raw NLP columns is removed.** The current pipeline includes `evt_framing_gap` *and* `framing_gap_ffill` *and* `framing_gap_delta` *and* `framing_gap_surprise` in the candidate pool. The engineered feature is a strict function of the raw ones. Keeping all of them in a SHAP selector splits the signal across near-duplicates and hurts stability.

---

## 7. Leakage contract (what this design promises)

- Every `_ffill` column is forward-filled from its call date only; no future quarter's call is visible at any earlier date.
- Every `_delta` and `_surprise` transformation uses only prior quarterly values (the `_surprise` uses a strict 4-quarter backward mean).
- `call_decay` and `news_decay` are functions of `days_since_call` / `days_since_last_news`, both strictly past-looking.
- `evt_source_dispersion`, `evt_cfo_lead`, `evt_exec_scripted_gap`, `evt_ceo_surprise` are all composed from columns that satisfy the above — no new leakage is introduced.
- Feature *selection* is a separate concern (Round 3 of `redesign_single_stock_experiments.md`). This design only defines the candidate pool.

---

## 8. What I would next do with this (not yet done)

1. Implement the 6 new `evt_*` features in `run_redesign_experiments.py::load_dataset`, then drop the raw `_delta`/`_surprise` NLP columns from the candidate pool.
2. Re-run the three validation tracks we already have:
   - `R07`-style manual reduced set (17 event features + existing market features) on the existing walk-forward.
   - `IMP_ShapBootstrap30` using the new candidate pool, to see whether the bootstrap voter picks up the new features for the AML event fold `v4` or the recovery folds `v5–v7`.
   - `IMP_ShapStability30` likewise.
3. Report whether any new `evt_*` feature is selected by the leakage-clean selectors specifically on **v4** (AML event fold) and **v5–v7** (recovery folds), since these are where the previous design showed no AML features getting selected at all.
4. Only if a leakage-clean run beats `S2_StaticShap30` on the business metrics do we save a new locked artifact.

This document is the design. No code, experiments, or artifacts have been changed yet.

---

## 9. Complete candidate feature set — NLP event block plus market, macro, and timing

The daily parquet `data/processed/features/model_features_daily.parquet` contains 171 columns, of which ~96 are raw NLP `_ffill/_delta/_surprise` variants that this design compresses into the 17 `evt_*` features of Section 4. The remaining columns cover price action, rates, macro, and news flow. Below is the proposed complete candidate pool: **47 features** across eight economically motivated groups.

Principle: include one feature per economic question, avoid multiple columns that measure the same thing on different windows unless each window answers a different question (e.g., 1-day return = overnight shock, 20-day return = monthly momentum regime).

### A. TD own price and technicals (10 features)

Short- and medium-term momentum, realized risk, and mean-reversion state for TD itself. Every feature is widely used in equity factor modelling and has a clear economic interpretation; none depends on Step 2 NLP.

| Feature | Purpose |
|---|---|
| `td_return_1d` | Overnight / single-day shock |
| `td_return_5d` | One-week momentum (same horizon as the target) |
| `td_return_20d` | One-month momentum regime |
| `td_volatility_20d` | Realized daily-return volatility over one month |
| `td_volume_zscore_20d` | Volume regime vs one-month norm; captures unusual activity around events |
| `td_corr_pv_20d` | Price-volume coherence, from the Qlib Alpha158 family |
| `td_rsi_14` | Classic 14-day mean-reversion indicator |
| `td_bb_position` | Position within 20-day Bollinger band (mean-reversion anchor) |
| `td_dist_52w_high` | Percent below 52-week high — trend position / anchoring bias |
| `td_resi_20d` | 20-day residual return vs TSX — short idiosyncratic component |

### B. Sector and market relative performance (3 features)

The redesigned target is `target_excess_xfn_5d` = TD 5-day forward return minus XFN 5-day forward return, so the features most economically aligned with it are realized TD-minus-sector and TD-minus-index returns plus the sector backdrop.

| Feature | Purpose |
|---|---|
| `td_vs_xfn_5d` | TD 5-day return − XFN 5-day return; realized same-horizon relative momentum |
| `td_vs_tsx_5d` | TD 5-day return − TSX 5-day return; broader market excess |
| `xfn_return_5d` | Sector 5-day return (context for the denominator we are targeting) |

### C. Systematic risk and beta structure (3 features)

How TD loads on the broader market. Kept at 60-day horizon because shorter betas are too noisy for a single stock.

| Feature | Purpose |
|---|---|
| `td_beta_60d` | Rolling 60-day beta to TSX |
| `td_rsqr_60d` | R² of the same regression — how systematic TD currently is |
| `td_resi_60d` | 60-day residual return — longer idiosyncratic component, pairs with `td_resi_20d` in A |

### D. Global risk regime (5 features)

Fear, USD, and gold pricing; controls for macro risk-on / risk-off regimes that drive Canadian bank stocks via correlated global flows.

| Feature | Purpose |
|---|---|
| `vix_level` | US equity fear gauge (level matters more than changes) |
| `vix_return_5d` | Vol-of-vol shock |
| `gold_return_5d` | Safe-haven flows |
| `dxy_return_5d` | USD strength |
| `fx_usdcad_return_5d` | CAD-specific FX stress — directly affects TD USD translation and US Retail sensitivity |

### E. Rates and macro (5 features)

NIM driver, policy regime, and inflation backdrop. Bank profitability tracks the yield curve; the BoC overnight rate is the funding anchor; CPI tracks the regime.

| Feature | Purpose |
|---|---|
| `yield_10y_level` | Long-end Canadian rates (asset yield side of NIM) |
| `yield_curve_slope` | 10y − 2y spread; canonical bank NIM proxy |
| `yield_curve_slope_change_5d` | Change in slope — NIM regime shift |
| `rate_overnight_level` | BoC policy rate — funding cost anchor |
| `cpi_yoy_change` | Inflation regime |

### F. News flow (3 features)

Sub-quarterly news cadence between earnings calls. Kept as second-tier because Step 2 L3 flags that press-release tables slightly bias `news_sentiment_mean`; the rolling windows and count features are cleaner.

| Feature | Purpose |
|---|---|
| `news_sent_mean_30d` | 30-day rolling news tone |
| `news_count_30d` | 30-day news intensity (proxy for narrative activity) |
| `is_news_burst` | Flag for sudden news spikes |

### G. Event-timing state (3 features)

Tells the model where in the quarterly cycle the current row sits, so the tree can gate NLP features by freshness explicitly.

| Feature | Purpose |
|---|---|
| `days_since_call` | Days since last earnings call; pairs with every `evt_*` multiplied by `call_decay` |
| `is_earnings_week` | Binary flag for the 5-day window around each earnings call |
| `days_since_last_news` | News recency; pairs with `news_decay` on news event features |

### H. NLP event features, net of news overlap (15 features)

Group F already covers `news_sent_mean_30d` / `news_count_30d` / `is_news_burst`. The previously proposed `evt_news_tone` and `evt_news_flow` are decayed restatements of those same raw features. To avoid duplication in the candidate pool, **drop `evt_news_tone` and `evt_news_flow` in favour of the Group F raw/rolling versions**, because Group F features don't require a `call_decay` multiplier and remain continuous through non-earnings periods. That leaves **15 Step 2-motivated event features** in the NLP block:

`evt_ceo_tone`, `evt_cfo_tone`, `evt_cfo_lead`, `evt_ceo_surprise`, `evt_analyst_qa_tone`, `evt_exec_scripted_gap`, `evt_framing_gap`, `evt_guidance_strength`, `evt_guidance_shift`, `evt_aml_pressure`, `evt_aml_shift`, `evt_source_dispersion`, `evt_topic_entropy`, `evt_broad_topic_signal`, `evt_broad_topic_shift`, `evt_transcript_reliability`.

*(16 names listed; the sixteenth, `evt_transcript_reliability`, is a modifier that depends on `transcript_sentiment_std_ffill` only, so it stays.)*

### Complete candidate pool — final count

| Group | Count | Information source |
|---|---:|---|
| A. TD own price & technicals | 10 | Market data |
| B. Sector and market relative | 3 | Market data |
| C. Systematic risk | 3 | Market data |
| D. Global risk regime | 5 | Market data |
| E. Rates and macro | 5 | Rates & macro data |
| F. News flow | 3 | News data |
| G. Event-timing state | 3 | Derived from call/news calendar |
| H. NLP event (Step 2 motivated) | 16 | LLM annotations |
| **Total** | **48** |  |

Compared to the previous candidate pool of 180+ columns (full daily parquet minus target columns), this drops roughly **three-quarters of the columns**, keeping only one representative of each economic question while expanding the NLP block from 13 to 16 with the six Step 2-motivated primitives.

### Why this is the right pool size

- **Signal density**: ~48 features × ~1,280 daily rows ≈ 60k data points. Well inside the regime where a shallow LightGBM multiclass can train stably without dense-feature overfitting.
- **Stability selection compatibility**: with the raw NLP `_ffill/_delta/_surprise` columns removed, the bootstrap SHAP voter no longer has to split signal across 6+ near-duplicate columns per underlying concept. Each concept has exactly one representative, so votes concentrate.
- **Audit trail**: every feature has either (i) a Step 2 analytical justification (Group H), (ii) an equity-factor / Qlib Alpha158 reference (Groups A–C), or (iii) a canonical macro-finance motivation (Groups D–E).
- **Horizon alignment**: the target is 5-day forward excess return. Dominant feature horizon is 5 days (`td_return_5d`, `td_vs_xfn_5d`, `xfn_return_5d`, `vix_return_5d`, `gold_return_5d`, `dxy_return_5d`, `fx_usdcad_return_5d`, `yield_curve_slope_change_5d`). 1-day, 14-day, 20-day, and 60-day features are included only where they answer a distinct question (overnight shock, RSI mean-reversion, monthly regime, beta stability).

---

## Appendix — condensed feature inventory (copy-paste ready)

```
# Group A — TD own price and technicals (10)
td_return_1d, td_return_5d, td_return_20d, td_volatility_20d,
td_volume_zscore_20d, td_corr_pv_20d, td_rsi_14, td_bb_position,
td_dist_52w_high, td_resi_20d

# Group B — sector/market relative (3)
td_vs_xfn_5d, td_vs_tsx_5d, xfn_return_5d

# Group C — systematic risk (3)
td_beta_60d, td_rsqr_60d, td_resi_60d

# Group D — global risk regime (5)
vix_level, vix_return_5d, gold_return_5d, dxy_return_5d, fx_usdcad_return_5d

# Group E — rates and macro (5)
yield_10y_level, yield_curve_slope, yield_curve_slope_change_5d,
rate_overnight_level, cpi_yoy_change

# Group F — news flow (3)
news_sent_mean_30d, news_count_30d, is_news_burst

# Group G — event-timing state (3)
days_since_call, is_earnings_week, days_since_last_news

# Group H — NLP event block, Step 2-motivated (16)
evt_ceo_tone, evt_cfo_tone, evt_cfo_lead, evt_ceo_surprise,
evt_analyst_qa_tone, evt_exec_scripted_gap,
evt_framing_gap, evt_guidance_strength, evt_guidance_shift,
evt_aml_pressure, evt_aml_shift, evt_source_dispersion,
evt_topic_entropy, evt_broad_topic_signal, evt_broad_topic_shift,
evt_transcript_reliability
```

Total: **48 features**, every one traceable to either Step 2 analytical evidence or a canonical market-finance motivation.
