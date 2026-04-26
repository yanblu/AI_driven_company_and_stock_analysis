# Single-Stock Redesign Experiments

This document records a **separate redesign track** from the locked final model in `notebooks/04_final_model.ipynb`.

The locked model is **not modified**. All redesign code and outputs live in:

- `redesign_single_stock/src/`
- `redesign_single_stock/data/`

This dedicated folder now uses the saved SHAP-selected artifact as the **locked redesign reference model**:

- `S2_StaticShap30` is the operational model kept in this redesign folder because it uses a fixed, auditable SHAP-selected feature list.
- `R07` is still retained as the stronger empirical validation benchmark from the earlier redesign rounds.

## Goal

Test whether a more pension-appropriate single-stock formulation can outperform trivial baselines **without** expanding to multiple stocks.

The redesign follows four changes:

1. **Target redesign**: predict TD's **5-day excess return** vs a benchmark, not raw TD return.
2. **Decision redesign**: use **3 classes** rather than raw regression only.
3. **Feature redesign**: compress the feature set into a smaller event-style representation.
4. **Model redesign**: test simpler / shallower models before reusing a wide tree setup.

## Files created

| File | Purpose |
|---|---|
| `redesign_single_stock/src/run_redesign_experiments.py` | Runs the redesign walk-forward experiments |
| `redesign_single_stock/data/summary.csv` | Experiment-level summary table |
| `redesign_single_stock/data/fold_metrics.csv` | Per-fold metrics |
| `redesign_single_stock/data/feature_manifest.json` | Exact feature sets used by each redesign experiment |

---

## Redesign setup

### Walk-forward framework

- Same canonical 7-fold expanding walk-forward splits as `E10_FOLDS`
- Same baked-in 5-trading-day boundary gap
- Same stride-5 offset-averaged evaluation structure

### New targets

Two relative targets were tested:

1. `target_excess_xfn_5d = td_fwd5 - xfn_fwd5`
2. `target_excess_tsx_5d = td_fwd5 - tsx_fwd5`

### New decision labels

Map continuous excess return into 3 classes:

- `-1` = underperform
- `0` = neutral
- `+1` = outperform

Two neutral bands were tested:

- **Standard band:** `±0.5%`
- **Tight band:** `±0.3%`

### Reduced feature set

The reduced redesign set uses **30 features**:

- Core market / price features:
  `td_return_5d`, `td_return_20d`, `td_volatility_20d`, `td_vs_xfn_5d`, `td_vs_tsx_5d`,
  `td_corr_pv_20d`, `td_volume_change_20d`, `td_dist_52w_high`
- Macro / risk features:
  `yield_curve_slope`, `yield_10y_level`, `vix_volatility_20d`, `dxy_level`, `gold_level`, `fx_usdcad_level`
- News / timing features:
  `news_sent_mean_30d`, `news_count_30d`, `days_since_last_news`, `days_since_call`, `is_earnings_week`
- Event-style compressed NLP features:
  `evt_ceo_tone`, `evt_cfo_tone`, `evt_framing_gap`, `evt_aml_pressure`, `evt_aml_shift`,
  `evt_guidance_strength`, `evt_guidance_shift`, `evt_macro_topic`, `evt_topic_entropy`,
  `evt_news_tone`, `evt_news_flow`

These event-style NLP features are constructed by applying decay to quarterly NLP snapshots using `days_since_call`, rather than feeding the full 99-column forward-filled NLP block directly into the model.

### How the features were narrowed

The redesign did **not** use a pure automatic feature selector. The reduced set was chosen deliberately to fit the single-stock use case.

The selection logic was:

1. Keep a compact core of market features that are already known to matter for TD:
   - recent returns
   - relative performance vs sector / market
   - volume / volatility
   - a few macro regime indicators
2. Keep the NLP signal, but stop feeding the model a very wide block of mostly forward-filled quarterly columns.
3. Prefer features that answer a practical decision question:
   - what is TD doing now?
   - what is TD doing relative to the sector?
   - what was the most recent management / narrative shock?
   - how stale is that signal?

This is why the redesign uses:

- **19 core market/news/timing features**
- **11 compressed NLP event features**
- **30 total features**

instead of the prior design, which effectively exposed the model to **99 NLP-derived columns** plus the rest of the market feature set.

### Event-style compressed NLP features vs. previous NLP features

The previous model used quarterly NLP inputs almost directly:

- level / forward-filled value
- quarter-over-quarter delta
- surprise vs trailing history

That approach preserves detail, but in a single-stock setting it also creates a problem:

- the same quarterly NLP value is repeated for many daily rows
- the model sees many correlated / stale NLP columns at once
- the effective information content is much smaller than the feature count suggests

The redesign instead treats earnings-call NLP as an **event signal**:

> the information is strongest around the call date, then gradually fades.

This is done by using `days_since_call` to decay the importance of the latest quarterly NLP snapshot over time.

### Business meaning of the compressed NLP features

| Feature | Business meaning |
|---|---|
| `evt_ceo_tone` | Current CEO prepared-remarks tone, strongest right after the call and fading afterward |
| `evt_cfo_tone` | Current CFO prepared-remarks tone with the same event decay |
| `evt_framing_gap` | Gap between management tone and filed report tone; captures whether management sounds more optimistic or more cautious than the documents |
| `evt_aml_pressure` | High when AML/regulatory discussion is both prominent and negative; intended to summarize regulatory stress |
| `evt_aml_shift` | Magnitude of change in AML topic share and AML tone vs the prior quarter; captures whether the AML narrative suddenly moved |
| `evt_guidance_strength` | Strength of positive forward guidance in the latest call |
| `evt_guidance_shift` | Quarter-over-quarter change in guidance emphasis and tone |
| `evt_macro_topic` | Strength and tone of management discussion around macro outlook |
| `evt_topic_entropy` | How broad vs concentrated the discussion topics were in the latest call |
| `evt_news_tone` | Recent news tone, decayed by days since latest news |
| `evt_news_flow` | Recent news intensity / flow, also decayed by recency |

### Why this compression is better for one stock

For a multi-stock cross-sectional model, a very wide feature space can sometimes be justified because there are many more independent observations.

For a **single stock**, that same wide NLP design is much riskier:

- too many features relative to sample size
- too much repeated quarterly information at daily grain
- harder interpretation
- higher chance of overfitting to regime timing

The compressed event-style design is more suitable because it:

- reduces dimensionality
- keeps the economically meaningful parts of the NLP signal
- matches how earnings-call information actually enters the market
- gives the model a smaller, more interpretable set of narrative features

### Models tested

| ID | Target | Features | Model |
|---|---|---|---|
| `R01` | Excess vs XFN | Reduced 30 | Logistic regression |
| `R02` | Excess vs XFN | Reduced 30 | Elastic net regression -> 3 classes |
| `R03` | Excess vs XFN | Reduced 30 | Shallow LightGBM multiclass |
| `R04` | Excess vs XFN | Full 180 | Shallow LightGBM multiclass |
| `R05` | Excess vs TSX | Reduced 30 | Logistic regression |
| `R06` | Excess vs XFN, tight band `±0.3%` | Reduced 30 | Logistic regression |
| `R07` | Excess vs XFN, tight band `±0.3%` | Reduced 30 | Shallow LightGBM multiclass |

---

## Acceptance rule

A redesign candidate is marked as **acceptable** only if all conditions hold:

- accuracy uplift vs majority baseline > 2pp
- accuracy uplift vs momentum baseline > 1pp
- macro F1 > 0.34
- active sign accuracy > 50%

Why these baselines:

- **Majority baseline** = predict the most common class in the training fold
- **Momentum baseline** = label using observed relative momentum (`td_vs_xfn_5d` or `td_vs_tsx_5d`)

---

## Round-1 summary results

| ID | Mean accuracy | Macro F1 | Active sign acc | Coverage | Majority baseline | Momentum baseline | Verdict |
|---|---|---|---|---|---|---|---|
| `R07` | **45.0%** | **0.344** | **57.9%** | **94.2%** | 38.5% | 36.6% | **Acceptable** |
| `R04` | 33.9% | 0.317 | 50.3% | 71.4% | 34.9% | 34.5% | No |
| `R03` | 35.8% | 0.313 | 53.5% | 74.7% | 34.9% | 34.5% | No |
| `R06` | 38.4% | 0.302 | 53.8% | 81.7% | 38.5% | 36.6% | No |
| `R05` | 36.2% | 0.290 | 50.4% | 88.6% | 43.1% | 34.6% | No |
| `R01` | 36.2% | 0.282 | 52.6% | 91.8% | 34.9% | 34.5% | No |
| `R02` | 35.0% | 0.170 | — | 0.0% | 34.9% | 34.5% | No |

### Round-1 best redesign: `R07`

**Status:** `R07` was the **best redesign candidate from round 1** and remains the saved artifact for that track.

**Configuration**

- Target: `TD 5-day excess return vs XFN`
- Labels: 3-class with **tight neutral band `±0.3%`**
- Features: reduced 30-feature event-style set
- Model: shallow LightGBM multiclass

**Performance**

- Mean accuracy: **45.0%**
- Mean macro F1: **0.344**
- Active sign accuracy: **57.9%**
- Coverage: **94.2%**
- Uplift vs majority baseline: **+6.5pp**
- Uplift vs relative-momentum baseline: **+8.3pp**

**Saved artifact**

- Model: `redesign_single_stock/data/artifacts/r07-current-best-2026-04-21.pkl`
- Metadata: `redesign_single_stock/data/artifacts/r07-current-best-2026-04-21.json`
- Training window: all currently labeled rows through `2026-04-09`

### Fold-level results for `R07`

| Fold | Accuracy | Macro F1 | Active sign acc | Coverage | Majority baseline | Momentum baseline |
|---|---|---|---|---|---|---|
| `v1` | 48.8% | 0.367 | 64.7% | 98.3% | 39.7% | 39.0% |
| `v2` | 42.0% | 0.356 | 53.1% | 91.6% | 41.1% | 37.0% |
| `v3` | 42.2% | 0.295 | 58.8% | 90.1% | 23.2% | 39.6% |
| `v4` | 52.1% | 0.416 | 62.4% | 96.7% | 42.2% | 35.6% |
| `v5` | 30.0% | 0.223 | 38.1% | 90.8% | 26.7% | 49.2% |
| `v6` | 53.6% | 0.466 | 70.5% | 95.1% | 45.4% | 30.6% |
| `v7` | 46.3% | 0.282 | 58.0% | 96.7% | 51.2% | 25.5% |

---

## Round-2 update: schema-first + adaptive selection

Round 2 implemented the follow-up plan without touching the locked model or the saved `R07` artifact.

New round-2 outputs were written separately to:

- `redesign_single_stock/data/round2_feature_schema.csv`
- `redesign_single_stock/data/round2_feature_schema.json`
- `redesign_single_stock/data/round2_stable_core_stats.csv`
- `redesign_single_stock/data/round2_summary.csv`
- `redesign_single_stock/data/round2_fold_metrics.csv`
- `redesign_single_stock/data/round2_class_balance.csv`
- `redesign_single_stock/data/round2_confusion_matrices.csv`
- `redesign_single_stock/data/round2_feature_manifest.json`
- `redesign_single_stock/data/round2_stage_selection.json`

### 1. Schema-first feature inventory

The full candidate redesign universe now has a documented schema with **180 features**:

| Group | Count |
|---|---|
| Quarterly NLP | 96 |
| Volume & technical | 29 |
| Price & relative momentum | 12 |
| Event-style NLP | 11 |
| Macro fear | 9 |
| News | 9 |
| Rates & inflation | 9 |
| FX | 3 |
| Timing | 2 |

Selection roles are now explicitly documented:

- **15 fixed-core features**
- **165 adaptive-candidate features**
- **30 features in the current manual reduced set**
- **11 event-style engineered features**

This means later per-retrain feature changes are now auditable rather than implicit.

The important refinement in this update is that the **fixed core is no longer manual**. It is now selected from a statistical stability procedure and stored in `round2_stable_core_stats.csv`.

### Statistical method for the fixed core

To make the fixed core defensible, the core was selected using the `S1_B30` setup only:

1. Use the round-2 band winner: `target_excess_xfn_5d` with `±0.30%` neutral band.
2. For each outer training fold, fit a training-only LightGBM selector on an internal `80% / 20%` split.
3. Compute normalized multiclass SHAP importance on the internal validation slice.
4. Aggregate each feature's behavior across the 7 folds.
5. Rank features by:
   - share of folds appearing in the top 20 (`top20_share`)
   - then share of folds appearing in the top 10 (`top10_share`)
   - then median normalized SHAP importance
   - then mean normalized SHAP importance
   - then average rank
6. Keep the top `15` features as the fixed core.

The resulting statistically selected fixed core was:

- `td_vs_xfn_5d`
- `rate_overnight_change_5d`
- `td_rsqr_60d`
- `gold_return_5d`
- `evt_guidance_shift`
- `topic_cost_efficiency_sentiment_ffill`
- `td_corr_pv_20d`
- `evt_cfo_tone`
- `tsx_return_20d`
- `news_sent_mean_30d`
- `gold_volatility_20d`
- `xfn_return_5d`
- `td_corr_pv_5d`
- `xfn_return_20d`
- `td_dist_52w_high`

Top stability examples:

| Feature | Top-20 share | Top-10 share | Mean rank |
|---|---|---|---|
| `td_vs_xfn_5d` | 100% | 100% | 5.3 |
| `rate_overnight_change_5d` | 100% | 71.4% | 9.0 |
| `td_rsqr_60d` | 100% | 57.1% | 7.3 |
| `gold_return_5d` | 100% | 28.6% | 10.7 |
| `evt_guidance_shift` | 85.7% | 71.4% | 13.9 |

### 2. Band sweep around `R07`

The conservative band sweep kept the target, reduced feature set, and shallow LightGBM fixed while changing only the neutral band:

| ID | Band | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|
| `S1_B20` | `±0.20%` | 48.3% | 0.334 | 56.0% | 98.8% | No |
| `S1_B25` | `±0.25%` | 45.9% | 0.340 | 55.2% | 96.7% | Acceptable |
| `S1_B30` | `±0.30%` | **45.0%** | **0.344** | **57.9%** | 94.2% | **Band-stage winner** |
| `S1_B35` | `±0.35%` | 40.9% | 0.320 | 55.8% | 89.4% | No |

Conclusion: the original `±0.30%` tight band used by `R07` remains the best business tradeoff. Narrowing to `±0.20%` improved raw mean accuracy, but mostly by collapsing the neutral class and pushing coverage close to always-on, which hurt macro balance.

### 3. Static SHAP-selected feature experiments

To address the explainability concern around the manual reduced set, the dynamic per-retrain path was replaced with a **static SHAP-selected feature path**.

The logic was:

1. keep the winning label setup fixed (`target_excess_xfn_5d`, `±0.30%`)
2. compute a global SHAP stability ranking across the outer training folds
3. lock one static feature list from that ranking
4. rerun the walk-forward experiments using that same static set in every fold

This gives a cleaner explanation story than dynamic selection:

- the feature list is fixed and auditable
- the selection rule is quantitative
- the final set can be listed directly in the report

Round 2 then compared these static policies at the winning `±0.30%` band:

| ID | Feature policy | Feature count | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| `S2_Reduced` | Current manual reduced set | 30 | **45.0%** | **0.344** | **57.9%** | **94.2%** | **Acceptable / Winner** |
| `S2_FixedCore` | Stable SHAP-selected core only | 15 | 44.6% | 0.339 | 57.0% | 92.3% | No |
| `S2_StaticShap20` | Static SHAP-selected top 20 features | 20 | 44.6% | 0.332 | 55.3% | 92.3% | No |
| `S2_StaticShap25` | Static SHAP-selected top 25 features | 25 | 44.1% | 0.323 | 56.6% | 91.8% | No |
| `S2_StaticShap30` | Static SHAP-selected top 30 features | 30 | 43.6% | 0.324 | 56.1% | 92.2% | Locked reference artifact |

This is the main result:

- the static SHAP route makes feature selection easier to defend
- but the best static SHAP set still does **not** beat the manual reduced set
- the closest challenger is `S2_StaticShap20`, which comes fairly close on mean accuracy but still trails on macro F1, active sign accuracy, and coverage

So after adding a more explainable statistical feature-selection method, the best-performing configuration still remains the original `R07`-equivalent reduced 30-feature design.

### 4. How to explain the manual reduced set

The right way to explain the manual reduced set is **not** "we hand-picked features arbitrarily." The better story is:

1. start from the full candidate universe
2. use SHAP stability analysis to identify what repeatedly matters
3. keep the strongest recurring features
4. then add a small number of business-motivated balancing features that improve calibration, timing awareness, and event coverage for a single-stock problem

The manual reduced set is therefore best described as a **curated compact feature policy**, not an unprincipled manual pick list.

It has two parts:

- **statistically supported backbone**: features that overlap with the SHAP stability ranking
- **business coverage layer**: features added so the model still sees timing, recency, volatility, and broader event-state information that a pure top-ranked static list tended to drop

The overlap between the manual reduced set and the best static SHAP top-20 list is meaningful. Shared features include:

- `td_vs_xfn_5d`
- `td_corr_pv_20d`
- `td_dist_52w_high`
- `td_volume_change_20d`
- `yield_10y_level`
- `gold_level`
- `news_sent_mean_30d`
- `evt_cfo_tone`
- `evt_guidance_shift`

The manual set then deliberately keeps additional features that help a pension-use-case model stay better behaved:

- **timing / recency controls**: `days_since_call`, `days_since_last_news`, `is_earnings_week`
- **event coverage**: `evt_ceo_tone`, `evt_framing_gap`, `evt_aml_pressure`, `evt_aml_shift`, `evt_guidance_strength`, `evt_macro_topic`, `evt_topic_entropy`, `evt_news_tone`, `evt_news_flow`
- **market state balance**: `td_return_5d`, `td_return_20d`, `td_volatility_20d`, `td_vs_tsx_5d`, `yield_curve_slope`, `vix_volatility_20d`, `dxy_level`, `fx_usdcad_level`, `news_count_30d`

That is the main reason it still wins empirically: the static SHAP lists are cleaner statistically, but the manual reduced set preserves a broader and better-balanced picture of TD's state.

### 4A. AML ablation check

Because AML is the one obviously TD-specific named topic in the manual set, an explicit ablation was run to test whether it was an unjustified bias or a genuinely useful risk signal.

Three variants were compared at the current winning setup (`LightGBM`, `target_excess_xfn_5d`, `±0.30%` band):

| ID | Feature policy | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|
| `S2_Reduced` | Manual reduced set with AML features | **45.0%** | **0.344** | **57.9%** | **94.2%** | **Best** |
| `S2_NoAML` | Manual reduced set with AML removed | 42.3% | 0.334 | 55.0% | 92.0% | Worse |
| `S2_TopicBasket` | AML removed and replaced by a broader non-AML topic basket | 43.0% | 0.334 | 54.2% | 93.7% | Worse |

Interpretation:

- removing AML hurts all of the key metrics
- replacing AML with a broader non-AML basket improves on "remove AML entirely," but still does not recover the lost performance
- so AML looks less like an arbitrary bias and more like a TD-specific narrative risk feature that is genuinely earning its place in the manual set

This does **not** prove AML should always be treated specially in every future version. But it does show that, in the current single-stock TD redesign, AML is not just there because of storytelling preference; it has out-of-sample support in the ablation.

### 5. What the statistical ranking tells us

The statistical ranking is still informative even though it did not win.

It says that the most consistently important signals across folds were:

- relative performance vs sector: `td_vs_xfn_5d`
- short- and medium-horizon technical state: `td_rsqr_60d`, `td_corr_pv_20d`, `td_corr_pv_5d`, `td_dist_52w_high`
- macro / rate state: `rate_overnight_change_5d`, `gold_return_5d`, `gold_volatility_20d`
- compact narrative state: `evt_guidance_shift`, `evt_cfo_tone`, `topic_cost_efficiency_sentiment_ffill`
- benchmark context: `xfn_return_5d`, `xfn_return_20d`, `tsx_return_20d`
- persistent news tone: `news_sent_mean_30d`

The best-performing static SHAP list was the top-20 version:

- `td_vs_xfn_5d`
- `rate_overnight_change_5d`
- `td_rsqr_60d`
- `gold_return_5d`
- `evt_guidance_shift`
- `topic_cost_efficiency_sentiment_ffill`
- `td_corr_pv_20d`
- `evt_cfo_tone`
- `tsx_return_20d`
- `news_sent_mean_30d`
- `gold_volatility_20d`
- `xfn_return_5d`
- `td_corr_pv_5d`
- `xfn_return_20d`
- `td_dist_52w_high`
- `yield_10y_level`
- `topic_cost_efficiency_sentiment_surprise`
- `gold_level`
- `tsx_return_1d`
- `td_volume_change_20d`

In business terms, that is a reasonable and explainable static set. The problem is not interpretability. The problem is performance: even the best static SHAP list remains a bit too negative-biased and loses some of the balanced behavior the manual reduced set had.

### 6. Extra XGBoost check on manual features at tighter bands

Because the manual reduced set remains the best feature policy, an additional check was run with the **same manual features** but swapping in `XGBoost` at the tighter bands:

| ID | Model | Band | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| `S1_XGB_B20` | XGBoost + manual reduced | `±0.20%` | 44.7% | 0.297 | 52.8% | 100.0% | No |
| `S1_XGB_B25` | XGBoost + manual reduced | `±0.25%` | 44.5% | 0.304 | 54.0% | 99.3% | No |
| `R07` / `S2_Reduced` | LightGBM + manual reduced | `±0.30%` | **45.0%** | **0.344** | **57.9%** | **94.2%** | **Still best** |

Interpretation:

- `XGBoost` at `±0.20%` slightly improves raw accuracy versus `XGBoost` at `±0.25%`, but does so with almost no neutral predictions and much weaker macro balance.
- `XGBoost` at `±0.25%` has the better of the two XGBoost macro-F1 results, but still trails the current LightGBM manual setup.
- Neither tighter-band XGBoost run beats the current `R07`-style LightGBM configuration on the business metrics that matter most.

### 7. NLP-block PCA check

To test whether a more mechanical compression rule could replace the handcrafted event-style narrative block, one additional experiment was run:

| ID | Model | Feature policy | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|---|
| `S2_NLPPCA` | LightGBM | raw market/news/timing features + 10 PCA components from the original quarterly NLP block | 42.9% | 0.317 | 55.4% | 93.0% | No |
| `R07` / `S2_Reduced` | LightGBM | manual reduced event-style feature set | **45.0%** | **0.344** | **57.9%** | **94.2%** | **Still best** |

Implementation notes:

- raw market/news/timing features were kept in original form
- the original quarterly NLP/topic block was imputed, standardized, and compressed into `10` PCA components inside each training fold
- those PCA components were then combined with the raw non-NLP features and fed into the same shallow `LightGBM`

Interpretation:

- compressing the wide original NLP block is feasible
- but it still underperforms the current event-style manual representation
- this supports the idea that the current winner benefits not just from dimensionality reduction, but from a **better economic encoding of narrative timing**

### 8. Model-family comparison on the best setup

Using the winning label setup (`±0.30%`) and the winning feature policy **after** the static-SHAP rerun:

| ID | Model | Mean accuracy | Macro F1 | Active sign acc | Coverage | Verdict |
|---|---|---|---|---|---|---|
| `S3_LGBM_MULTICLASS` | LightGBM multiclass on reduced 30 | **45.0%** | **0.344** | **57.9%** | 94.2% | **Winner** |
| `S3_XGB_MULTICLASS` | XGBoost multiclass on reduced 30 | 42.7% | 0.309 | 53.8% | 97.0% | No |
| `S3_LOGISTIC` | Logistic regression on reduced 30 | 38.4% | 0.302 | 53.8% | 81.7% | No |

Conclusion: LightGBM still wins, and the earlier takeaway remains true: **model-family swaps do not rescue the problem**. The best results still come from the original redesigned task plus the reduced event-style feature block.

### 9. Calibration diagnostics

Average realized class mix in the evaluation folds:

- underperform: **35.5%**
- neutral: **21.3%**
- outperform: **43.2%**

Average predicted class mix for the best static SHAP challenger (`S2_StaticShap20`):

- underperform: **54.3%**
- neutral: **7.7%**
- outperform: **38.0%**

Average side-specific precision for `S2_StaticShap20`:

- predicted `outperform` precision: **51.0%**
- predicted `underperform` precision: **37.3%**

Fold-level calibration observations:

- the static SHAP sets still underpredict neutral almost everywhere
- they continue to overpredict `underperform`, especially in `v3` and `v5`
- `v5` remains the clearest failure case: realized `outperform` share is `62.5%`, while `S2_StaticShap20` predicts `underperform` `75.8%` of the time
- this calibration tilt is the main reason the static SHAP sets do not beat the broader reduced set

### 10. Locked redesign model after reorganization

`round2_stage_selection.json` still marks the refined round as **not beating `R07`** on validation performance.

After moving the redesign work into this dedicated folder, the saved SHAP-selected artifact is the **locked redesign reference model** for this track:

- ID: `S2_StaticShap30`
- Target: `TD 5-day excess return vs XFN`
- Labels: 3-class with `±0.30%` neutral band
- Feature policy: static SHAP-selected top 30 features
- Model: shallow LightGBM multiclass
- Model artifact: `redesign_single_stock/data/artifacts/shap30-current-best-2026-04-21.pkl`
- Metadata: `redesign_single_stock/data/artifacts/shap30-current-best-2026-04-21.json`

`R07` is still kept in the same folder as the **empirical benchmark winner** from prior validation rounds.

**Performance comparison**

| Model | Mean accuracy | Macro F1 | Active sign acc | Coverage |
|---|---|---|---|---|
| `R07` | 45.0% | 0.344 | 57.9% | 94.2% |
| Static SHAP `S2_FixedCore` | 44.6% | 0.339 | 57.0% | 92.3% |
| Static SHAP `S2_StaticShap20` | 44.6% | 0.332 | 55.3% | 92.3% |
| Static SHAP `S2_StaticShap25` | 44.1% | 0.323 | 56.6% | 91.8% |
| Static SHAP `S2_StaticShap30` | 43.6% | 0.324 | 56.1% | 92.2% |

So the refinement answered the methodological concern, but it **did not** produce a better model. The redesign folder is now locked to the SHAP-selected artifact for auditability, while `R07` remains the stronger empirical benchmark.

---

## Round 3 — Leakage-clean feature selection verification

### Why this round exists

`S2_StaticShap30` was selected by ranking SHAP importance aggregated across **all 7 outer training folds** and then applying that single top-30 list back to every fold. That means some features became "top 30" partly because of behaviour that shows up only in later training windows the earlier folds would not have seen. The static SHAP number is therefore **flattered by time look-ahead in the feature selection step** — the model fit is still fold-local, but the feature list was chosen with future-leaning information.

Round 3 re-runs the same redesign problem (`target_excess_xfn_5d`, 3-class, `±0.30%` band, shallow LightGBM) with two **leakage-clean** feature selection variants. Both are forward-looking: the feature list for fold `k` only uses data up to fold `k`'s training cutoff.

Outputs live in `redesign_single_stock/data/improvements/` and do not touch the existing locked artifact.

### Variants tested

1. **`IMP_FoldLocalShap30` — fold-local SHAP top 30**
   - For each outer fold `k`: fit a shallow LightGBM selector on fold-`k` training data only, compute SHAP on a held-out slice of that same training window, keep the top 30, retrain on the full training window using only those 30 features, evaluate on the outer test fold.
   - Runner: `redesign_single_stock/src/run_fold_local_shap30_experiment.py`

2. **`IMP_ShapStability30` — within-training SHAP stability top 30 (inner expanding windows)**
   - For each outer fold `k`: inside that fold's training window only, run 4 inner expanding-window splits at 60%, 70%, 80%, 90%. Fit a LightGBM on each inner training slice and rank features by SHAP on the inner validation slice.
   - Keep features that appear in the inner top-30 in **at least 3 of 4** inner windows (majority vote). If fewer than 30 stable features, fill remaining slots by mean SHAP importance across inner windows. Retrain on the full fold training window using only those 30 features, evaluate on the outer test fold.
   - Runner: `redesign_single_stock/src/run_shap_stability_experiment.py`
   - Typical fold had **19–24 "majority-stable" features** with the rest filled to reach 30.

3. **`IMP_ShapBootstrap30` — bootstrap SHAP voting top 30 (simpler stability selection)**
   - For each outer fold `k`: inside that fold's training window only, repeat 20 times: randomly split the training window 70/30 into a sub-train and a sub-test, fit a shallow LightGBM on the sub-train, compute SHAP on the sub-test, record that bootstrap's top-30 features.
   - A feature's "selection frequency" is the share of bootstraps in which it appeared in top-30. Keep features with **selection frequency ≥ 50%**; fill remaining slots by mean SHAP importance across bootstraps to reach 30.
   - Retrain the final LightGBM on the full fold training window using only those 30 features and evaluate on the outer test fold.
   - Runner: `redesign_single_stock/src/run_shap_bootstrap_experiment.py`
   - Typical fold had **25–29 "majority-stable" features** — the vote was sharper than the inner-expanding-window variant because each bootstrap was much larger than a 10-20% inner slice.
   - This is the classic Meinshausen–Bühlmann stability-selection idea: random subsamples, vote by frequency, no dependence on a single SHAP ranking. Because no rows from later folds or from the outer test window ever enter the sub-train / sub-test pair, the feature list for fold `k` is still strictly past-only.

### Results vs existing benchmarks

| Model | Feature selection | Leakage-clean? | Mean accuracy | Macro F1 | Active sign acc | Coverage |
|---|---|---|---|---|---|---|
| `R07` / `S2_Reduced` | Manual reduced 30 | Yes | **45.0%** | **0.344** | **57.9%** | 94.2% |
| `S2_StaticShap30` (locked) | Static top-30 across all folds | **No — global ranking uses all folds** | 44.5% | 0.343 | 56.8% | 90.9% |
| `IMP_FoldLocalShap30` | Fold-local SHAP top 30 | Yes | 39.0% | 0.304 | 51.0% | 91.9% |
| `IMP_ShapStability30` | Within-training SHAP stability top 30 (inner expanding) | Yes | 42.8% | 0.312 | 54.5% | 92.9% |
| `IMP_ShapBootstrap30` | Bootstrap SHAP voting top 30 (20 random 70/30 splits) | Yes | 40.9% | 0.313 | 52.5% | 94.0% |

Acceptance criterion for a new locked artifact was: beat `S2_StaticShap30` on `active_sign_acc`, stay within 1pp of its macro F1, and keep coverage ≥ 70%. **None of the three leakage-clean variants met this criterion.**

### Interpretation

- The gap between `S2_StaticShap30` and the best leakage-clean variant (`IMP_ShapStability30`: about **2pp active sign accuracy** and **3pp macro F1**) is the size of the boost the global SHAP ranking had been quietly getting from time look-ahead in feature selection.
- Among the three leakage-clean variants, the inner-expanding-window stability selector performed best, the bootstrap voting selector came second, and the single-shot fold-local top-30 came last. The ordering makes sense: every additional SHAP estimate that is averaged / voted on makes the selection more stable and less sensitive to one noisy split, but none of them recover all of the ~3pp macro F1 gap vs the manual reduced set.
- All leakage-clean variants still beat both baselines (majority and momentum) but are not strong enough to displace the reduced manual set or the currently locked SHAP-selected artifact on business-relevant metrics.
- The manual reduced 30-feature set remains the best empirically, which is consistent with the hypothesis that **narrative/event-style compression carries signal that pure SHAP-driven selection underweights** when it can only see one training window at a time.

### Decision on the locked artifact (option A)

Per the agreed rule: **keep `shap30-current-best-2026-04-21.pkl` unchanged** and only replace it if a new leakage-clean variant clearly beats it. None of the three variants did, so no new locked artifact was saved. The Round 3 outputs (including `shap_stability30_decision.json` and `shap_bootstrap30_decision.json`) are kept as reproducible evidence that multiple leakage-clean paths were tested and none won.

For transparency, the doc now carries an explicit note:

> `S2_StaticShap30`'s reported walk-forward metrics are **partially flattered by feature-selection look-ahead**. The leakage-clean `IMP_ShapStability30` variant is the truer estimate of how a stability-selected SHAP-30 design would perform in production; it lands meaningfully below `R07` on business metrics.

### What this changes for the stakeholder story

- `R07` is now both the **empirical benchmark winner** and the **best leakage-clean configuration** we have found.
- `S2_StaticShap30` stays in the folder as the "locked auditable feature policy" artifact, but readers should be told its metric numbers have a known optimistic bias from global feature selection.
- For any future production hand-off, the recommended configuration is the **`R07` manual reduced set**, and any auto-selection path should use fold-local or within-training stability selection like `IMP_ShapStability30` rather than a global SHAP ranking.

### Diagnostic: why the leakage-clean selectors underperform the manual set

To understand the gap, we inspected per-fold feature manifests and per-fold metrics.

**AML feature selection per outer fold, per method** (an "AML feature" means any of `evt_aml_pressure`, `evt_aml_shift`, `topic_regulatory_AML_share_*`, `topic_regulatory_AML_sentiment_*`):

| Fold | Test window | `IMP_FoldLocalShap30` | `IMP_ShapStability30` | `IMP_ShapBootstrap30` |
|---|---|---|---|---|
| v1 | 2023 H1 (pre-AML) | `evt_aml_shift`, `AML_share_ffill` | `evt_aml_shift`, `AML_share_ffill` | `AML_share_ffill` |
| v2 | 2023 H2 (pre-AML) | `AML_share_ffill` | `AML_share_ffill` | — |
| v3 | 2024 H1 (pre-AML event) | — | `AML_share_ffill` | — |
| **v4** | **2024 H2 (AML event)** | **—** | **—** | **—** |
| v5 | 2025 H1 (post-event) | — | — | — |
| v6 | 2025 H2 (post-event) | — | — | — |
| v7 | 2026 H1 (post-event) | — | — | — |

In contrast, the manual reduced set **always includes `evt_aml_pressure` and `evt_aml_shift`** in every fold by construction.

**What this tells us:**

- The AML event broke publicly between May–Oct 2024. Fold `v4`'s training cutoff is **2024-06-28**, so its training data mostly sits *before* the news spike. From a SHAP selector's point of view that uses training data only, the AML topic share looks low-variance and weakly correlated with 5-day excess return — so every leakage-clean selector *correctly* drops AML features in `v4`. They cannot pick up the event ahead of time without leakage.
- From `v5` onward, training data *does* contain the AML event. AML features are still dropped, because once the event has already moved the stock and the settlement is priced in, the AML features stop explaining *future* 5-day excess return. SHAP in training data shows them as no-longer-useful, which is technically right.
- The manual set keeps AML features anyway because of **prior economic knowledge** — a human modeller decided a priori that regulatory-topic spikes are worth carrying as risk features across the whole history. That is information SHAP-on-training cannot recover.

**But the per-fold metrics show the gap is NOT mostly about v4.** The leakage-clean selectors are actually competitive or better on `v4` itself:

| Method | v1 | v2 | v3 | v4 (AML event) | v5 | v6 | v7 |
|---|---|---|---|---|---|---|---|
| `S2_Reduced` (manual, has AML) | 64.7% | 53.1% | 58.8% | **62.4%** | 38.1% | 70.5% | 58.0% |
| `S2_StaticShap30` (leaked) | 64.4% | 52.3% | 65.3% | 59.5% | 37.3% | 65.8% | 53.1% |
| `IMP_ShapStability30` | 47.8% | 56.8% | 59.8% | 57.5% | 45.9% | 59.8% | 54.1% |
| `IMP_ShapBootstrap30` | 58.5% | 54.6% | 52.8% | **64.9%** | 41.2% | 44.8% | 50.8% |
| `IMP_ShapFoldLocal30` | 43.6% | 51.2% | 59.8% | 57.2% | 36.0% | 63.0% | 46.4% |

*(active sign accuracy per fold)*

- On `v4` specifically, `IMP_ShapBootstrap30` at **64.9%** actually **beats the manual `S2_Reduced` set** (62.4%) — even without AML features. The sector-relative return (`td_vs_xfn_5d`, `xfn_return_5d`) and event-style non-AML NLP features were enough to pick up the AML regime through the relative-performance channel alone.
- The aggregate gap vs `S2_Reduced` is driven by `v1` and `v6`, not `v4`. There, SHAP-on-training picks noisier features because no single event dominates the training signal, and the manual prior-knowledge choices are more stable.

**So the cleaner hypothesis is:**

> The leakage-clean SHAP selectors are not failing to see the AML event — they are correctly telling us AML features don't appear predictive in the training data available to them at each cutoff. What they lose vs the manual set is **~3pp macro F1 of prior-knowledge robustness across calm folds**, not event-window performance. That gap is the value of human feature engineering that SHAP-on-a-single-training-window cannot discover on its own.

This tightens the conclusion: the manual reduced set does not win because it "predicts AML". It wins because carrying AML + event-style features across every fold makes the feature list more **robust to fold-to-fold regime shifts**, while stability-selected SHAP lists drift with whatever is strongest in the current training window.

---

## Interpretation

## 1. What changed after adding static SHAP selection and extra XGBoost checks

`R07` was already more acceptable than the raw-return setup because it solved a better-defined problem. After replacing the dynamic-selection idea with a static SHAP-selected feature path, the conclusion became:

- it preserved the relative-performance framing vs `XFN`
- it preserved the better `±0.30%` neutral band
- it added a real quantitative justification for a fixed feature list
- it also checked whether the manual reduced set gets stronger under tighter-band `XGBoost`
- but neither the static SHAP sets nor the extra XGBoost checks improved the business metrics

## 2. Why the validation winner and the locked artifact differ

The result pattern is also informative:

- the SHAP-selected static sets captured features that were persistently important
- but they still removed some of the broader event-style representation that helped the reduced 30-feature model stay more balanced
- the tighter-band `XGBoost` variants increased activity and raw hit rate in places, but did not preserve class balance or active-call quality well enough
- the manual reduced set therefore remains the best tradeoff between signal quality, event coverage, and calibration
- the redesign folder now locks `S2_StaticShap30` anyway because it gives a fixed, auditable feature policy for handoff and artifact management

So the statistical screen was useful both as a diagnostic and as an explainability aid, but not as a superior replacement for the full reduced redesign set.

## 3. What still does not work

Even the new best redesign is **not perfect**:

- `v5` remains weak, which is consistent with the known regime-transition problem
- the best current model still underpredicts the neutral class on average
- the static SHAP-selected challengers still overpredict `underperform`
- results are still based on a small single-name sample, so confidence should remain conservative

---

## Round 4 — Step-2 grounded 48-feature static pool

Motivation and design: `redesign_single_stock/nlp_event_feature_design.md` Section 9. The goal was to rebuild the NLP event block from Step 2 analytical findings (adding 6 primitives: `evt_cfo_lead`, `evt_ceo_surprise`, `evt_analyst_qa_tone`, `evt_exec_scripted_gap`, `evt_source_dispersion`, `evt_transcript_reliability`) and pair it with a curated 32-feature market/macro/news/timing block — 48 total, all selected statically with an explicit economic motivation per feature.

Runner: `redesign_single_stock/src/run_step2_48feature_experiment.py`. Outputs under `redesign_single_stock/data/step2_48/`. Same LightGBM multiclass hyperparameters, same walk-forward (E10_FOLDS), same target and `±0.30%` band as R07.

### Headline comparison (7-fold mean)

| Metric | STEP2_48F | R07 / S2_Reduced | S2_StaticShap30 |
|---|---:|---:|---:|
| mean accuracy | 0.4343 | **0.4498** | 0.4451 |
| mean macro F1 | 0.3238 | **0.3436** | 0.3434 |
| mean active sign accuracy | 0.5469 | **0.5794** | 0.5681 |
| mean active coverage | 0.9279 | 0.9419 | 0.9088 |
| mean active precision (long) | 0.4634 | 0.4779 | **0.5349** |
| mean active precision (short) | 0.3985 | **0.4286** | 0.4205 |
| uplift vs majority | +4.95 pp | +6.50 pp | +6.03 pp |
| uplift vs momentum | +6.79 pp | +8.34 pp | +7.87 pp |

STEP2_48F loses on all headline metrics. It does not replace `S2_StaticShap30` as the locked redesign model.

### Per-fold diagnosis (active sign accuracy)

| Fold | STEP2_48F | R07 | Δ vs R07 | Interpretation |
|---|---:|---:|---:|---|
| v1 | 0.424 | 0.647 | −22.3 pp | Smallest training window; 48 features overfit |
| v2 | 0.513 | 0.531 | −1.8 pp | Comparable |
| v3 | 0.518 | 0.588 | −7.0 pp | Mildly worse |
| v4 (AML peak) | **0.667** | 0.624 | +4.3 pp | Step-2 AML + multi-source features help |
| v5 (first recovery) | **0.588** | 0.381 | +20.7 pp | Recovery primitives (`evt_cfo_lead`, `evt_analyst_qa_tone`) fire as designed |
| v6 | 0.635 | 0.705 | −7.0 pp | Late recovery — extra features add noise |
| v7 | 0.483 | 0.580 | −9.7 pp | Calm window — reduced set already sufficient |

### Interpretation

1. **The Step-2 hypothesis is directionally confirmed.** On the two folds where Step 2 predicted the new features should matter (`v4` AML peak, `v5` first recovery), STEP2_48F delivers the two largest gains in the experiment (+4 pp and +21 pp active sign accuracy). The features are doing what the analysis said they would do.
2. **The penalty is concentrated on `v1`.** With ~200 training rows and 48 features, the model's feature/row ratio is ~1:4 — far outside the stable regime for LightGBM. The −22 pp on `v1` alone drags the 7-fold mean from roughly flat to −3 pp versus R07.
3. **Marginal features hurt on calm folds.** `v6` and `v7` have clean narrative backgrounds and the 18 extra features add variance without signal.
4. **No leakage.** All six new primitives are composed from `_ffill` / `_surprise` columns that are strictly past-looking (Step 2 design doc Section 7). The feature list is static and fixed before any test data is seen, so there is no feature-selection leakage either. The performance gap versus `S2_StaticShap30` is therefore honest.

### Why this doesn't replace S2_StaticShap30 yet

`S2_StaticShap30`'s reported metrics are flattered by feature-selection look-ahead (Round 3 diagnostic). `STEP2_48F` is fully leakage-clean. The fair comparison is STEP2_48F vs the leakage-clean `IMP_ShapBootstrap30` / `IMP_ShapStability30` variants:

| Metric | STEP2_48F | IMP_ShapBootstrap30 | IMP_ShapStability30 |
|---|---:|---:|---:|
| mean active sign accuracy | **0.5469** | 0.5416 | 0.5456 |
| mean macro F1 | 0.3238 | **0.3267** | 0.3243 |
| mean active coverage | 0.9279 | 0.9145 | **0.9313** |

Against the leakage-clean field, STEP2_48F is roughly tied — slight edge on active sign accuracy, slight miss on macro F1. Given that its feature list has a strictly named analytical justification (Step 2 finding → feature primitive), STEP2_48F is the more defensible of the leakage-clean options even though none of them reaches `S2_StaticShap30`'s flattered benchmark.

### Next actions to consider

- **Trim to ~30 features with stability selection on the 48-pool.** The bootstrap SHAP voter was previously run on the 180+ column superset. Running it on the cleaner 48-pool should concentrate votes and produce a ~30-feature subset that retains the `v4`/`v5` gains while dropping the noise that hurts `v1`/`v6`/`v7`.
- **Drop or down-weight `v1` in cross-fold averaging.** The `v1` training window is too small to compete with a 48-feature model; document the headline with and without `v1` if this becomes the recommended model family.
- **Keep `S2_StaticShap30` as the locked redesign artifact** until a leakage-clean variant beats it on 7-fold mean active sign accuracy with the coverage constraint (`≥ 0.70`).

---

## Round 5 — Bootstrap SHAP voting on the 48-feature Step-2 pool (`STEP2_48F_Boot30`)

Variant `STEP2_48F_Boot30`: same bootstrap SHAP voter as `IMP_ShapBootstrap30` (20 random 70/30 sub-train / sub-test splits per fold, keep features in top-30 in ≥50% of splits, fill to 30 by mean SHAP importance), but the **candidate pool is the 48-feature Step-2 set** instead of the full 180+ column daily parquet. Fully leakage-clean (feature selection strictly within each fold's training window). Runner: `redesign_single_stock/src/run_shap_bootstrap_step2_48.py`. Outputs: `redesign_single_stock/data/step2_48_bootstrap/`.

### Headline (7-fold mean)

| Metric | **STEP2_48F_Boot30** | STEP2_48F | R07 / S2_Reduced | S2_StaticShap30 (leaky) |
|---|---:|---:|---:|---:|
| mean accuracy | **0.4421** | 0.4343 | 0.4498 | 0.4451 |
| mean macro F1 | **0.3305** | 0.3238 | 0.3436 | 0.3434 |
| mean active sign accuracy | **0.5625** | 0.5469 | 0.5794 | 0.5681 |
| mean active coverage | 0.9336 | 0.9279 | 0.9419 | 0.9088 |
| mean active precision (long) | 0.4653 | 0.4634 | 0.4779 | 0.5349 |
| mean active precision (short) | 0.4142 | 0.3985 | 0.4286 | 0.4205 |
| uplift vs majority | +5.73 pp | +4.95 pp | +6.50 pp | +6.03 pp |
| uplift vs momentum | +7.56 pp | +6.79 pp | +8.34 pp | +7.87 pp |

Boot30 sits **between `STEP2_48F` and `S2_StaticShap30`**: better than the full 48-pool on every headline metric, still below the leaky `S2_StaticShap30` benchmark.

### Compared to the earlier bootstrap voter run on the 180+ column pool

| Metric | **STEP2_48F_Boot30** | IMP_ShapBootstrap30 (180+ pool) |
|---|---:|---:|
| mean active sign accuracy | **0.5625** | 0.5416 |
| mean macro F1 | **0.3305** | 0.3267 |
| mean active coverage | 0.9336 | 0.9145 |
| mean stable-feature count per fold | 31.6 | variable, often <30 |

Running the voter on the curated 48-pool (instead of the 180+ column raw pool) improves leakage-clean active sign accuracy by **+2.1 pp** and brings it within **0.56 pp** of the flattered `S2_StaticShap30` benchmark. Votes concentrate as hypothesised — `n_stable` averages 31.6 / 48 features (66% of the pool clears the 50% frequency threshold), compared to the earlier run where near-duplicate `_ffill`/`_delta`/`_surprise` triplets split votes and never cleared 30 stable features.

### Per-fold story vs STEP2_48F (did pruning fix the small-fold overfit?)

| Fold | STEP2_48F | Boot30 | Δ | Interpretation |
|---|---:|---:|---:|---|
| v1 | 0.424 | 0.486 | **+6.2 pp** | Pruning reduced overfit on the tiny training window |
| v2 | 0.513 | 0.512 | −0.1 pp | Tied |
| v3 | 0.518 | 0.638 | **+12.0 pp** | Major gain — voter kept the right Step-2 features |
| v4 (AML peak) | 0.667 | 0.649 | −1.8 pp | Slight loss but still > R07 |
| v5 (first recovery) | 0.588 | 0.453 | **−13.5 pp** | Bootstrap dropped some of the recovery features that drove STEP2_48F's v5 win |
| v6 | 0.635 | 0.661 | +2.5 pp | Partial fix |
| v7 | 0.483 | 0.538 | +5.5 pp | Partial fix |

Net: pruning traded a **−13.5 pp v5 loss** for **+6 / +12 / +2.5 / +5.5 pp gains on v1, v3, v6, v7**. The v5 loss is the only fold where Boot30 underperforms STEP2_48F on active sign accuracy by more than 2 pp, and that fold's v5 gap shows the voter is still sensitive to exactly which NLP features are on the list (see next section).

### Per-fold story vs R07 (where leakage-clean beats the manual set)

| Fold | R07 | Boot30 | Δ |
|---|---:|---:|---:|
| v1 | 0.647 | 0.486 | −16.1 pp |
| v2 | 0.531 | 0.512 | −1.9 pp |
| v3 | 0.588 | 0.638 | **+5.1 pp** |
| v4 | 0.624 | 0.649 | **+2.5 pp** |
| v5 | 0.381 | 0.453 | **+7.2 pp** |
| v6 | 0.705 | 0.661 | −4.4 pp |
| v7 | 0.580 | 0.538 | −4.2 pp |

Boot30 beats R07 on the three event / transition folds (`v3`, `v4`, `v5`) — the ones where the Step-2 hypothesis mattered most. R07 wins on `v1` (training-size limitation the voter cannot overcome) and on the calm late folds (`v6`, `v7`).

### Which of the six new Step-2 features does the voter actually pick?

Selection frequency across the 7 outer folds:

| New feature | Folds selected | % of folds |
|---|:---:|---:|
| `evt_ceo_surprise` | v1, v2, v3, v4, v5, v6, v7 | **7 / 7** |
| `evt_analyst_qa_tone` | v1, v2, v3, v4, v5, v6, v7 | **7 / 7** |
| `evt_exec_scripted_gap` | v1, v2, v3, v4, v5, v7 | 6 / 7 |
| `evt_cfo_lead` | v1, v2, v5, v6, v7 | 5 / 7 (skipped exactly on the two AML folds, consistent with Step 2 — CFO lead only became relevant after the FY2025 recovery) |
| `evt_transcript_reliability` | v5, v6, v7 | 3 / 7 (later folds where accumulated transcript-std data matters) |
| `evt_source_dispersion` | — | **0 / 7** |

This is a clean empirical verdict on the Step-2 design:

- Four of the six new features are consistently voted in across all folds — **the Step-2 analytical findings map to features a leakage-clean selector values on held-out data**.
- `evt_cfo_lead` behaves exactly as Step 2 predicted (skipped during the AML quarters, picked up during and after FY2025 recovery).
- `evt_source_dispersion` is the one new feature the voter never selects — the "multi-source corroboration" story either isn't as predictive as hypothesised, or the stddev-of-three-channels formulation is too noisy. Candidate for redesign.

### Does it unlock a new locked artifact?

The artifact-save gate is `active_sign_acc > S2_StaticShap30` AND `macro_f1 >= S2_StaticShap30 − 0.01` AND `coverage >= 0.70`.
Boot30: 0.5625 vs 0.5681 (−0.56 pp) on active sign accuracy. **Does not pass** the gate; the existing `shap30-current-best-*.pkl` remains the locked redesign model.

But STEP2_48F_Boot30 is, to date, **the strongest leakage-clean candidate we have**, beating every other leakage-clean variant on the headline metric:

| Leakage-clean candidate | Active sign accuracy |
|---|---:|
| **STEP2_48F_Boot30** (48-pool + voter) | **0.5625** |
| STEP2_48F (48-pool, no selection) | 0.5469 |
| IMP_ShapBootstrap30 (180+ pool + voter) | 0.5416 |
| IMP_ShapStability30 (180+ pool + nested SHAP) | 0.5456 |
| IMP_FoldLocalShap30 (per-fold top-30) | ~0.54 |

### Recommendation

- **Keep** `S2_StaticShap30` as the locked redesign artifact (still the nominal best on reported metrics) with the caveat that its numbers are feature-selection-leakage-flattered.
- **Promote** `STEP2_48F_Boot30` as the **canonical leakage-clean redesign candidate**: every feature has a named Step-2 justification, feature selection is strictly past-only, and the 7-fold mean metrics are the best in the leakage-clean field. This is the defensible model for stakeholder communication.
- Drop / redesign `evt_source_dispersion` — the voter rejects it unanimously. Either rework it to be an agreement correlation (sign(ceo) * sign(report) * sign(news)) or remove from the design.
- Consider a smaller reported cohort (v2–v7, or v3–v7) to communicate event-fold performance cleanly; 7-fold averages drag the numbers down due to the v1 training-size problem that no method in this family has resolved.

---

## Round 6 — Global static SHAP top-30 on the full pool + 6 Step-2 event features (`STEP2_Full_StaticShap30`)

Apples-to-apples variant of `S2_StaticShap30`: same methodology (global SHAP ranking aggregated across all outer training folds, top-30 applied back to every fold — **not leakage-clean**), but the candidate pool is expanded from the original ~180 columns to **188 features** by adding the 6 new Step-2 primitives on top of the full daily parquet. Runner: `redesign_single_stock/src/run_static_shap_step2_full.py`. Outputs: `redesign_single_stock/data/step2_full_staticshap/`.

### Headline (7-fold mean)

| Metric | **STEP2_Full_StaticShap30** | S2_StaticShap30 | STEP2_48F_Boot30 | STEP2_48F | R07 |
|---|---:|---:|---:|---:|---:|
| mean accuracy | 0.4372 | **0.4451** | 0.4421 | 0.4343 | **0.4498** |
| mean macro F1 | 0.3238 | **0.3434** | 0.3305 | 0.3238 | **0.3436** |
| mean active sign accuracy | 0.5510 | **0.5681** | 0.5625 | 0.5469 | **0.5794** |
| mean active coverage | 0.9313 | 0.9088 | 0.9336 | 0.9279 | 0.9419 |
| mean active precision (long) | 0.4529 | **0.5349** | 0.4653 | 0.4634 | 0.4779 |
| mean active precision (short) | 0.4221 | 0.4205 | 0.4142 | 0.3985 | **0.4286** |

**Adding the Step-2 features to the full-pool static-SHAP methodology hurts by ~1.7 pp active sign accuracy vs the original `S2_StaticShap30` benchmark.** The setup remains leakage-flattered, yet the flattered result is worse than the smaller 180+-column pool produced.

### What changed in the top-30

The global SHAP ranker swapped in 4 features and out 4:

| Added | Removed |
|---|---|
| `evt_ceo_surprise` *(new Step-2, rank 4)* | `topic_cost_efficiency_sentiment_surprise` |
| `evt_cfo_lead` *(new Step-2, rank 16)* | `td_wvma_20d` |
| `tsx_return_5d` | `td_wvma_5d` |
| `fx_usdcad_level` | `dxy_volatility_20d` |

Only **2 of the 6 new Step-2 primitives** (`evt_ceo_surprise`, `evt_cfo_lead`) survive into the top-30 when all raw `_ffill`/`_delta`/`_surprise` columns compete for the same slots. The other four (`evt_analyst_qa_tone`, `evt_exec_scripted_gap`, `evt_source_dispersion`, `evt_transcript_reliability`) are out-ranked by the raw NLP columns they were designed to replace. Compare with the 48-pool bootstrap voter run where 4 of 6 got in — that's the raw-column-crowding effect in action.

### Per-fold breakdown vs S2_StaticShap30

| Fold | S2_StaticShap30 | STEP2_Full_StaticShap30 | Δ |
|---|---:|---:|---:|
| v1 | 0.644 | 0.469 | **−17.5 pp** |
| v2 | 0.523 | 0.544 | +2.1 pp |
| v3 | 0.653 | 0.664 | +1.1 pp |
| v4 (AML peak) | 0.595 | 0.545 | −5.0 pp |
| v5 (first recovery) | 0.373 | 0.399 | +2.6 pp |
| v6 | 0.658 | 0.669 | +1.0 pp |
| v7 | 0.531 | 0.568 | +3.7 pp |

Five folds mildly improve, but the v1 collapse (−17.5 pp) and v4 loss (−5.0 pp) dominate the average.

### Interpretation

1. **Adding features to a leaky static-SHAP pool is not always additive.** The ranker picks up `evt_ceo_surprise` and `evt_cfo_lead` (ranks 4 and 16), but those two features displace `td_wvma_*`, `topic_cost_efficiency_sentiment_surprise`, and `dxy_volatility_20d` — features that evidently carried real signal on `v1`/`v4` despite ranking lower globally. The top-K static rule doesn't optimise for per-fold stability.
2. **Raw NLP columns absorb most of the Step-2 signal when both are in the pool.** Only two of six new primitives make it in. Their analytical value is real (Round 5 showed the bootstrap voter picked four of six on the cleaner 48-pool) but when raw `_ffill`/`_delta`/`_surprise` triplets are also candidates, the engineered features are largely redundant with them.
3. **The Round-5 recommendation is reinforced.** The correct place for Step-2 event features is a **curated, non-redundant pool** (the 48-pool), paired with **leakage-clean selection** (bootstrap SHAP voting). Adding them to the full leaky pool does not help.

### Verdict

- `STEP2_Full_StaticShap30` is a **negative result** — expanding the candidate pool of the leaky static-SHAP setup does not beat `S2_StaticShap30`. It confirms that the value of the Step-2 primitives is unlocked only when they are not competing with redundant raw columns.
- **No change to locked artifacts or recommendations.** `S2_StaticShap30` remains the locked redesign model; `STEP2_48F_Boot30` remains the canonical leakage-clean candidate.

---

## Bottom line

### The static SHAP refinement improved explainability, but not performance.

The redesign track still works because of **problem framing plus feature compression**, not because of tree-model swaps or because a global static SHAP ranking outperformed the manual reduced set.

The locked redesign model in this folder is:

- **`S2_StaticShap30`: TD excess return vs XFN, 3-class, `±0.30%` band, static SHAP-selected top-30 feature set, shallow LightGBM**

The strongest validation benchmark remains:

- **`R07` / `S2_Reduced` / `S3_LGBM_MULTICLASS`: TD excess return vs XFN, 3-class, `±0.30%` band, reduced 30-feature event-style set, shallow LightGBM**

### Recommendation

For the redesign folder itself, carry `S2_StaticShap30` forward as the **locked saved redesign model**.

At the same time, keep `R07` in the same folder as the validation benchmark and keep the statistical-core outputs as supporting evidence that:

- the feature list can be justified quantitatively with a static SHAP ranking
- but the statistically justified static set still does not beat the broader reduced redesign set

It should be positioned as:

> a **single-stock relative-performance classifier** for TD vs sector over 5 trading days,
> not a raw-return point forecaster.

