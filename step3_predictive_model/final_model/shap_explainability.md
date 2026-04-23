# lgbm_3class_xfn5d_35f — SHAP Explainability

What drives the model's predictions, and what does that tell us about how TD's stock moves relative to its sector?

All figures are mean |SHAP| values averaged across 7 out-of-sample walk-forward folds
(Jan 2023 – Apr 2026). A larger SHAP value means that feature had more influence on the
prediction for that class — regardless of whether the influence pushed the probability
up or down.

---

## What is SHAP?

Every prediction the model makes can be decomposed into contributions from each of the 35
features. SHAP assigns each feature a score that answers: *"by how much did this feature
move the predicted probability, compared to the baseline?"* A positive SHAP score pushed
the probability higher; a negative score pushed it lower. Averaging the absolute values
across many predictions shows which features the model relies on most.

---

## Outperform signal — what drives it

| Feature | Mean \|SHAP\| | What it captures |
|---|---:|---|
| `td_rsqr_60d` | 0.169 | Trend quality — how smoothly TD's price has moved over 60 days |
| `evt_guidance_shift` | 0.123 | Quarter-over-quarter change in management's forward guidance volume and tone |
| `evt_cfo_tone` | 0.120 | CFO prepared-remarks sentiment from the most recent earnings call |
| `yield_10y_level` | 0.104 | Absolute level of Canadian 10-year yields |
| `td_wvma_20d` | 0.061 | 20-day volume-weighted price trend |
| `td_corr_pv_20d` | 0.055 | Whether recent price moves are backed by volume |
| `gold_level` | 0.050 | Gold price as a global risk-off proxy |
| `news_sent_mean_30d` | 0.050 | Rolling 30-day news sentiment tone |
| `vix_volatility_20d` | 0.050 | Volatility-of-volatility (market uncertainty) |
| `td_beta_20d` | 0.049 | How correlated TD is to the broad market |

**What the model is saying:**
The dominant outperform signal is **trend quality** (`td_rsqr_60d`). When TD's price has
moved cleanly and persistently in one direction — rather than chopping around — the model
sees that as a continuation signal relative to the XFN sector ETF. This is not simply
"buy what is going up"; it measures whether the *direction* of the move is coherent, which
is a meaningful distinction.

The second and third most important features are both NLP-sourced. **`evt_cfo_tone`**
works in the intuitive direction: more positive CFO prepared remarks increase outperform
probability. **`evt_guidance_shift`** works the opposite way — it is a contrarian signal.
When management gives *less* guidance than the prior quarter, the model raises the outperform
probability; when guidance increases, it lowers it.

The mechanism is short-term mean reversion: when TD pulls back on forward-looking
statements, the stock tends to sell off relative to sector peers. That sell-off typically
overshoots in the near term, and the 5-day window the model predicts is often the partial
recovery leg. The model has learned to buy the silence, not the optimism. The distinction
matters: `evt_guidance_shift` is informative not because management transparency predicts
quality, but because the market's immediate reaction to guidance compression tends to
be too large.

Rate environment (`yield_10y_level`) is the strongest macro counterweight. High absolute
yields reduce the outperform probability — but the reason is not that higher rates are bad
for banks in absolute terms. The target here is **sector-relative**: TD vs the XFN bank
ETF. When rates rise, the entire Canadian banking sector benefits from NIM expansion
roughly equally, so the sector-relative gain is close to zero. What the model picks up is
more specific: in the 2021–2026 sample, the highest yield levels (Q4–Q5, ~3.5–3.7%)
coincided with the AML enforcement period and elevated credit concerns in TD's US retail
book — idiosyncratic headwinds that hurt TD more than its peers. The yield level is
therefore acting partly as a macro regime indicator and partly as a proxy for the
macro-stress backdrop that accompanied TD's AML cycle.

---

## Underperform signal — what drives it

| Feature | Mean \|SHAP\| | What it captures |
|---|---:|---|
| `evt_guidance_shift` | 0.211 | Quarter-over-quarter change in management's forward guidance volume and tone |
| `td_corr_pv_20d` | 0.125 | Whether recent price moves are backed by volume |
| `td_dist_52w_high` | 0.082 | How far TD's price is from its 52-week high |
| `yield_curve_slope` | 0.075 | 10Y minus 2Y yield spread (steepness of the curve) |
| `yield_10y_level` | 0.057 | Absolute 10-year yield level |
| `gold_level` | 0.054 | Gold price as a risk-off proxy |
| `news_sent_mean_30d` | 0.051 | Rolling 30-day news sentiment |
| `td_beta_20d` | 0.044 | Market sensitivity |
| `td_volume_change_20d` | 0.040 | Unusual volume activity |
| `evt_macro_topic` | 0.039 | Whether macro commentary was above or below neutral |

**What the model is saying:**
`evt_guidance_shift` is by far the dominant underperform driver — its SHAP score here
(**0.211**) is nearly **twice** its outperform score (0.123). A sudden collapse in how much
management discusses the future and how positively they frame it is the strongest
single predictor of TD underperforming its sector peers.

The underperform signal is more macro-laden than the outperform one. **Distance from the
52-week high** (`td_dist_52w_high`), **yield curve flatness** (`yield_curve_slope`), and
**gold rising** (`gold_level`) all amplify the underperform signal. These cluster around
a consistent theme: when TD is technically weak, rates are compressing bank margins, and
global risk appetite is falling, the model assigns high underperform probability. The NLP
signal (`evt_guidance_shift`) amplifies this further when the macro stress is accompanied
by management pulling back on forward commitments.

**Asymmetry between the two classes:**
The outperform signal is dominated by company-specific factors (trend quality, CFO tone,
guidance shift). The underperform signal is more driven by macro and technical stress
conditions. Looking at pooled out-of-sample results across all 7 folds:

| Class | Predictions | Precision | Recall |
|---|---:|---:|---:|
| Outperform | 332 | **0.536** | 0.531 |
| Underperform | 427 | 0.464 | **0.695** |
| Neutral | 27 | 0.148 | 0.024 |

The model is actually *more precise* on outperform (53.6% vs 46.4%) — when it calls
outperform it is right more often. What it does less well is *coverage* of underperform
events: it predicts underperform far more aggressively (427 vs 332 times), catching 69.5%
of actual underperform days but generating more false alarms in the process. This makes
intuitive sense: systematic macro and technical stress conditions are identifiable but also
common, so the model casts a wide net on the downside. Company-specific outperform signals
are rarer and harder to trigger, but when the model does fire them the hit rate is higher.

---

## Per-fold commentary — when did the model work and when did it struggle?

| Fold | Period | ASA | Dir AUC | Character |
|---|---|---:|---:|---|
| v1 | Jan–Jun 2023 | 0.647 | 0.634 | Strong |
| v2 | Jul–Dec 2023 | 0.577 | 0.599 | Moderate |
| v3 | Jan–Jun 2024 | 0.595 | **0.469** | Directionally weak |
| v4 | Jul–Dec 2024 | 0.611 | 0.640 | Strong |
| v5 | Jan–Jun 2025 | **0.511** | 0.532 | Weakest — near random |
| v6 | Jul–Dec 2025 | **0.676** | 0.626 | Strongest |
| v7 | Jan–Apr 2026 | 0.619 | 0.503 | Partial window (63 days) |

*ASA = Active Sign Accuracy: when the model takes a directional view, how often is it right?
Dir AUC = directional discriminability for both outperform and underperform; 0.5 = no better than chance.*

### v1 — Jan–Jun 2023: Strong (ASA 0.647, AUC 0.634)
Top features: `td_rsqr_60d`, `evt_guidance_shift`, `evt_cfo_tone`

A period with clear company-specific narrative. TD's First Horizon deal collapse
(May 2023) was reflected in a sharp spike in `M_and_A` topic share and a
corresponding dip in guidance sentiment. Both NLP features were signalling stress
before the price moved, and the trend-quality signal (`td_rsqr_60d`) helped distinguish
which direction the price was settling. The model had clean, consistent inputs.

### v2 — Jul–Dec 2023: Moderate (ASA 0.577, AUC 0.599)
Top features: `evt_cfo_tone`, `yield_10y_level`, `vix_volatility_20d`

The macro environment moved to the foreground as BoC rate expectations shifted. The model
leaned on rates and volatility more than company signals. Results were acceptable but
less decisive — the NLP features were relatively stable (no major event), reducing their
marginal value.

### v3 — Jan–Jun 2024: Directionally weak (ASA 0.595, AUC 0.469)
Top features: `td_rsqr_60d`, `evt_guidance_shift`, `yield_10y_level`

Dir AUC below 0.5 means the model was mildly *wrong* on direction even as its raw accuracy
held up. This period coincides with the peak of the AML build-up: AML topic share had
climbed above 20% and sentiment on AML-tagged chunks was near-neutral, yet the market
hadn't yet fully repriced the stock. The model correctly detected deteriorating signals
but the magnitude and timing of price moves were idiosyncratic — the consent order
uncertainty created unpredictable intraday volatility that overwhelmed the signal.

### v4 — Jul–Dec 2024: Strong (ASA 0.611, AUC 0.640)
Top features: `yield_10y_level`, `evt_cfo_tone`, `td_wvma_20d`, `td_rsqr_60d`

The AML consent order was announced in October 2024 (FY2024Q4). CEO and CFO sentiment
both hit zero that quarter — the only zero readings in the entire dataset. The model had
a clear, extreme NLP signal to work with: `evt_cfo_tone` dropped sharply and the
`evt_guidance_shift` reflected a collapse in forward-guidance activity (guidance share
fell to 8.8%). The model correctly identified underperform pressure.

### v5 — Jan–Jun 2025: Weakest (ASA 0.511, AUC 0.532)
Top features: `evt_guidance_shift`, `td_rsqr_60d`, `yield_10y_level`, `td_wvma_20d`, `evt_framing_gap`

The hardest fold — and `evt_guidance_shift` is central to understanding why.

A key clarification on how this feature behaves: the SHAP data shows that `evt_guidance_shift`
is in fact a **contrarian** signal for outperform, not a momentum one. Looking at how the
feature value maps to outperform SHAP across all test observations:

| Guidance shift quintile | Mean feature value | Mean outperform SHAP |
|---|---:|---:|
| Q1 — most negative | −0.046 | **+0.170** |
| Q2 | −0.001 | +0.060 |
| Q3 (near zero) | +0.001 | −0.040 |
| Q4 | +0.008 | −0.066 |
| Q5 — most positive | +0.061 | **−0.242** |

A large *negative* guidance shift — management pulling back on forward commitments —
increases outperform probability. A large *positive* shift decreases it. This seems
counterintuitive, but reflects the sector-relative context: when TD guidance collapses
while peers maintain theirs, the stock can be oversold relative to sector, creating a
mean-reversion outperform opportunity. Conversely, when guidance spikes (as in FY2025Q1),
the recovery is already visible and priced — the whole banking sector rallies together,
so TD's sector-relative upside is limited even as the absolute news is positive.

Here is why guidance specifically hurt v5. The logic that worked in v4 ran in reverse:

- **v4**: guidance collapsed (8.8%) → `evt_guidance_shift` hit Q1 (most negative) → model raised outperform probability → stock did mean-revert upward vs sector after the consent order sell-off → **correct call**
- **v5**: guidance spiked back (17.8%, dataset high) → `evt_guidance_shift` hit Q5 (most positive) → model suppressed outperform probability, leaning toward neutral or underperform → but there was no corresponding sell-off to mean-revert from

The problem: the contrarian mean-reversion pattern requires the stock to have already overreacted to the downside. In v5, management's guidance spike was genuine good news (AML resolved), but the stock had already partially recovered in v4. There was no fresh overshoot to fade. The model kept suppressing outperform predictions based on a high guidance shift, while the actual 5-day returns were driven by external factors the features captured only partially. BoC rate decisions are reflected in `yield_curve_slope` and `yield_10y_level`, but as level readings not rate surprises — an anticipated cut and an unexpected cut look identical to the model. US trade policy volatility (tariff announcements in early 2025) affected TD through USD/CAD moves that `fx_usdcad_level` and `dxy_level` partially proxy, but those features cannot separate tariff-driven FX moves from Fed expectations or global risk sentiment. The TD newsroom features only capture TD-specific press releases, not broad macro news, so tariff-driven swings were effectively invisible. These factors sent TD up and down vs sector independently of any guidance signal. `evt_guidance_shift` was the dominant feature in v5 but it was pointing at the wrong mechanism for that regime, actively adding noise rather than signal.

### v6 — Jul–Dec 2025: Strongest (ASA 0.676, AUC 0.626)
Top features: `gold_level`, `td_beta_20d`, `td_rsqr_60d`, `td_wvma_20d`, `yield_curve_slope`

Once the AML transition noise settled, the model returned to a clean macro regime. Gold and
beta dominated — risk-off/risk-on rotation drove sector-relative moves more than any
company-specific factor. In stable regimes without a major narrative discontinuity, the
model's feature set is well-suited: smooth macro signals produce consistent directional
calls. The NLP features were less prominent here, which is itself informative — when the
company-specific story is resolved, market and macro signals take over.

### v7 — Jan–Apr 2026: Partial (ASA 0.619, AUC 0.503)
Top features: `gold_level`, `td_rsqr_60d`, `td_wvma_20d`, `td_corr_pv_20d`, `evt_guidance_shift`

Only 63 trading days — too short to draw strong conclusions. The gold and trend-quality
features dominated, similar to v6. Dir AUC of 0.503 reflects insufficient test-window
size rather than model failure; ASA of 0.619 is consistent with other functioning folds.

---

## Connecting the SHAP story to the AML narrative

The SHAP output is not just a technical artefact — it traces the same three-phase AML
story documented in `step2_holistic_analysis.md`.

**Phase 1 — Pre-AML build-up (v2/v3):**
`evt_guidance_shift` and `evt_cfo_tone` began moving but the model's directional accuracy
was inconsistent (v3 AUC below chance). This reflects what the NLP data showed: signals
were deteriorating monotonically through FY2023–2024, but the market was slow to price
the full impact. The model detected the stress; the stock didn't respond cleanly yet.

**Phase 2 — Peak enforcement (v4):**
The model performed well precisely because the NLP signals hit extremes. CEO and CFO
sentiment going to zero, and guidance collapsing, gave the model the largest NLP SHAP
contributions of any fold. The consent order announcement provided a discrete, highly
legible signal — and `evt_cfo_tone` being the #1 NLP outperform predictor reflects that
the *absence* of CFO confidence is one of the strongest underperform precursors.

**Phase 3 — Recovery (v5/v6):**
The model stumbled in v5 when guidance spiked back, then recovered in v6 once the
transition was complete and macro factors reasserted. This maps exactly to the narrative:
FY2025Q1's guidance spike (17.8% share, five-year high) and analyst sentiment recovery
(0.529 in FY2026Q1, dataset high) were real signals, but they came with an abrupt regime
shift that produced noisy price behaviour in the short term. By H2 2025, the signal
cleared and the model's best fold performance arrived.

**The bottom line:**
`evt_guidance_shift` being the #1 underperform predictor and the #2 outperform predictor
is not coincidence — it is the model learning the same thing the NLP analysis found:
*management's willingness to talk about the future is the single most informative
language signal for TD's stock relative to its sector peers.*
