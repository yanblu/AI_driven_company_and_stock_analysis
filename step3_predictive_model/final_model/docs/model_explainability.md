# Model Explainability

SHAP is a technique that measures how much each input feature contributed to a given prediction. The observations below are drawn from pooled out-of-sample data across all 13 evaluation quarters. Beeswarm and walk-forward importance plots are in `notebooks/xgb_ensemble_v2_performance.ipynb` (Sections 8–9).

Each model produces separate signals for the **+1** (TD outperforms XFN) and **−1** (TD underperforms XFN) predictions. Where a feature drives both directions symmetrically, the two are covered in a single point.

---

## Model 1 — Price

### Observations

- **`td_dist_52w_high_v2`**
  - **What it measures**: How far TD's stock is from its 52-week high. Near zero = trading at its annual peak; more negative = deeper drawdown.
  - **What we observe**: TD near its 52-week high → model calls outperformance. TD well below it → model calls underperformance. The pattern is symmetric and this is the model's single strongest signal.
  - **What this suggests**: The model treats 52-week proximity as a momentum regime indicator. Stocks near their annual peak tend to keep outperforming; stocks in prolonged drawdown tend to keep lagging the sector.

- **`td_return_20d`**
  - **What it measures**: TD's total price return over the past month (20 trading days).
  - **What we observe**: A strong positive month → model leans outperformance. A flat or negative month → model leans underperformance. Notably, losses trigger a stronger reaction than equivalent gains — the model is more sensitive to recent weakness than recent strength.
  - **What this suggests**: Momentum matters in both directions, but a losing month is a more reliable warning than a winning month is a bullish signal. The model has learned that downside momentum is stickier than upside momentum for this stock.

- **`td_volatility_20d`**
  - **What it measures**: How erratically TD's stock has been moving day-to-day over the past month.
  - **What we observe**: High, choppy volatility → model tilts toward underperformance. Calm, stable price action → mild tilt toward outperformance. The downside reaction is noticeably stronger than the upside — this is primarily a risk alarm, not a balanced two-way signal.
  - **What this suggests**: Turbulent price action is a warning sign on its own. Stability, however, is not enough to call outperformance — the model still needs positive momentum signals alongside it.

---

## Model 2 — Transcript

### Observations

- **`days_since_call`**
  - **What it measures**: How many trading days have passed since TD's last earnings call.
  - **What we observe**: This feature doesn't point toward outperformance or underperformance on its own — it controls how much the other transcript signals matter. Fresh call (recent weeks) → model trusts and acts on transcript signals heavily. Stale call (60+ days ago) → model discounts all transcript signals and drifts toward neutral.
  - **What this suggests**: Call freshness is the on/off switch for the entire transcript model. The model learned on its own that old earnings calls stop being useful — no explicit rule was needed.

- **`transcript_analyst_qa_sentiment_mean_ffill`**
  - **What it measures**: The overall tone of the analyst Q&A session on the earnings call — whether the back-and-forth between analysts and management was constructive or tense.
  - **What we observe**: Positive Q&A → model calls outperformance. Tense or skeptical Q&A → model calls underperformance. The signal is symmetric and works with roughly equal strength in both directions.
  - **What this suggests**: When experienced analysts engage positively, the model reads it as external validation. When they push back, it treats this as an early warning — analyst scrutiny tends to precede price weakness.

- **`framing_gap_ffill`**
  - **What it measures**: Whether management sounds more optimistic in the earnings call than they do in their formal written filings. A positive gap means they are more upbeat verbally than on paper.
  - **What we observe**: Positive gap → model leans outperformance. No gap or alignment between verbal and written tone → little signal in either direction. Primarily an upside signal; it does not reliably predict downside.
  - **What this suggests**: When management speaks more confidently than their filings indicate, the model treats this as a leading indicator — conviction building before it has been formally communicated to the market.

---

## Model 3 — News

### Observations

- **`news_sent_mean_30d`**
  - **What it measures**: The average tone of all TD news coverage over the past 30 days — positive or negative on the whole. A single headline doesn't move this; it requires a sustained shift.
  - **What we observe**: A month of broadly positive coverage → model calls outperformance. A month of broadly negative coverage → model calls underperformance. Positive sentiment is a stronger outperformance signal than negative sentiment is an underperformance signal — at extremes of negativity, the model can occasionally flip to outperformance as the stock becomes oversold.
  - **What this suggests**: Sustained media narrative carries forward into price. Positive is more reliable than negative as a directional predictor; prolonged bad press sometimes overshoots and sets up a recovery rather than further decline.

- **`news_count_30d`**
  - **What it measures**: How many news articles about TD appeared in the last 30 days — regardless of whether they were positive or negative. Volume only, tone-blind.
  - **What we observe**: High article volume → strong underperformance signal. Quiet coverage → mild outperformance lean. The downside signal is significantly stronger than the upside, making this primarily a risk alarm.
  - **What this suggests**: News bursts cluster around bad events — regulatory developments, analyst downgrades, earnings concerns. Heavy media attention signals elevated scrutiny and uncertainty, not just activity. Quiet periods offer only mild reassurance.

---

## Per-Fold Deep Dive

The model's 13 evaluation quarters fall into three groups: **5 strong quarters** (65–70% directional accuracy), **2 significantly bad quarters** (below 40%, worse than a coin flip), and 6 moderate quarters near chance. Exact fold dates are in `model_choice.md`. Per-fold SHAP breakdowns for all three models are in `notebooks/xgb_ensemble_v2_performance.ipynb` (Section 10).

### When the model performs well

Strong quarters consistently share the same condition: **TD's stock is moving on its own story**, not dragged by sector-wide macro forces. In those environments:

- The **price model** leans on how close TD is trading to its 52-week high (`td_dist_52w_high_v2`) — a proxy for whether the stock is in a healthy momentum regime. In quarters where price trends are particularly clean, recent absolute return (`td_return_20d`) takes over as the lead signal. Either way, the model is reading clear directional momentum in TD's own price.

- The **transcript model** gates everything through `days_since_call` — it leads the transcript model in every strong fold without exception. What comes second shifted over time. In the 2023 strong folds (Folds 2 and 4), analyst Q&A sentiment (`transcript_analyst_qa_sentiment_mean_ffill`) and framing tone were the differentiating signals. From late 2024 onward (Folds 7, 10, 12), AML engagement (`topic_regulatory_AML_share_ffill`) moved to the front as TD's regulatory case took centre stage on earnings calls. The pattern — call freshness gates the signal, then a topic-specific feature takes over — is consistent; which topic dominates depends on what was in focus that quarter.

- The **news model** leans on sustained 30-day sentiment (`news_sent_mean_30d`) for outperformance and article volume (`news_count_30d`) for underperformance. It is most reliable when coverage is driven by TD-specific events rather than macro noise.

In short: the model performs well when fundamental signals — momentum, management tone, media narrative — are actually driving the stock. When one model has a clear signal and the others are quiet, the ensemble still works. It does not require all three to agree.

### Fold 1 — Mar–Jun 2023 (DA: 0.361) — SVB banking crisis

**What the model was using**: the price model was relying on recent price momentum (`td_return_20d`) to call outperformance — but that momentum had already been reversed by the crisis. The transcript model's top signal was `topic_credit_quality_share_ffill`, which it was using correctly as a warning: high credit quality discussion was pushing it toward underperformance, not outperformance (the beeswarm confirms all SHAP values were negative for the +1 class in this fold).

**What happened**: Silicon Valley Bank collapsed in March 2023, triggering sector-wide panic. TD and XFN fell together, making the sector-relative return (the model's actual target) noisy and effectively random. The model's directional calls — in both directions — had no reliable anchor.

**Why the model missed it**: the model had never seen a systemic banking crisis in its two-year training window. The price model was calling outperformance based on momentum that evaporated overnight. The transcript model's credit quality warning signal was valid in isolation, but TD's sector-relative return during a panic is not driven by what management said on a call — it's driven by how fearful the market is. The two signals were answering different questions. Only the news model's article volume (`news_count_30d`) flagged elevated systemic risk, but was outvoted.

### Fold 8 — Dec 2024–Mar 2025 (DA: 0.388) — US tariff shock

**What the model was using**: TD's stock had fallen well below its 52-week high (bearish price regime), executive tone on the call was cautious, analyst Q&A sentiment was negative, and news volume was elevated. All three models converged on underperformance — predicting −1 on 51 of 63 test days.

**What happened**: US tariff escalation in early 2025. The model called underperformance consistently. But TD actually **outperformed XFN on 51 of 63 days** — the model got the sector-relative direction almost completely backwards. The confusion matrix shows 40 days where the model called −1 and the actual result was +1.

**Why the model missed it**: TD, as a large Canadian retail bank, was more defensively positioned than many other XFN constituents — insurance companies, asset managers, and smaller financial firms that were more directly exposed to tariff-driven market disruption. As the broader selloff hit the XFN basket harder than TD's deposit-heavy business, TD's sector-relative return turned positive. The model had no feature to capture differential tariff sensitivity within the XFN basket. All of its signals (bearish price regime, cautious tone, high news volume) were reading TD-specific weakness correctly — but XFN fell further, flipping the sector-relative outcome.

---

## Walk-Forward Stability

Feature rankings are broadly stable across folds for the transcript and news models. `days_since_call` consistently leads the transcript model in every fold; `news_sent_mean_30d` leads the news model in most folds.

The price model shows more quarter-to-quarter variation, particularly between `td_dist_52w_high_v2` and `td_vs_xfn_20d` in the outperformance class. This is expected: momentum regimes shift (trending vs. mean-reverting periods), and the model relearns which price feature is most relevant each quarter.

---

## Key Investment Takeaways

- **AML discussion is a positive signal, not a negative one** — high AML topic share on the earnings call predicts outperformance, not underperformance. The model interprets active regulatory engagement as management working toward resolution. This warrants qualitative validation against specific quarters to confirm the direction holds.
- **Tone is what triggers downside calls, not topic** — executives can discuss credit quality, AML, or macro risk without triggering a negative signal if they do so confidently. It is cautious, muted, or defensive tone that drives underperformance predictions.
- **News volume is a risk indicator, independent of sentiment** — a surge in 30-day article count reliably predicts underperformance, even if individual articles are not negative. Media scrutiny itself is the signal.
- **The model's edge is strongest when TD is in focus** — in macro-driven quarters (banking sector stress, tariff shocks), the model loses its edge because company-specific features cannot capture market-wide moves.
- **Call freshness is the transcript model's biggest lever** — every transcript signal decays meaningfully as the quarter ages. The highest-value period for transcript-based signals is the first three to four weeks after an earnings call.
