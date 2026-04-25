# Model Explainability

SHAP measures how much each input feature pushed a given prediction up or down. The observations below come from out-of-sample data across all 13 evaluation quarters. Beeswarm and walk-forward importance plots are in `notebooks/xgb_ensemble_v2_performance.ipynb` (Sections 8–9).

Each model produces separate signals for **+1** (TD outperforms XFN) and **−1** (TD underperforms XFN). Where a feature drives both directions symmetrically, the two are covered in a single point.

---

## Model 1 — Price

### Observations

- `td_dist_52w_high_v2`
  - **What it measures**: How far TD's stock is from its 52-week high. Near zero = at its annual peak; more negative = deeper drawdown.
  - **What we observe**: TD near its 52-week high → model calls outperformance. TD well below it → model calls underperformance. This is the model's single strongest signal.
  - **What this suggests**: The model reads 52-week proximity as a momentum indicator. Stocks near their peak tend to keep outperforming; stocks in a prolonged drawdown tend to keep lagging.
- `td_return_20d`
  - **What it measures**: TD's total price return over the past month (20 trading days).
  - **What we observe**: A strong positive month → model leans outperformance. A flat or negative month → model leans underperformance. Losses trigger a stronger reaction than equivalent gains.
  - **What this suggests**: Recent weakness is a more reliable warning than recent strength is a bullish signal. Downside momentum is stickier than upside momentum.
- `td_volatility_20d`
  - **What it measures**: How erratically TD's stock has been moving day-to-day over the past month.
  - **What we observe**: High volatility → model tilts toward underperformance. Calm price action → mild tilt toward outperformance. The downside reaction is noticeably stronger — this is primarily a risk alarm.
  - **What this suggests**: Turbulent price action is a warning sign on its own. Stability alone is not enough to call outperformance — the model still needs positive momentum alongside it.

---

## Model 2 — Transcript

### Observations

- `days_since_call`
  - **What it measures**: How many trading days have passed since TD's last earnings call.
  - **What we observe**: This feature does not point toward outperformance or underperformance on its own — it controls how much the other transcript signals matter. Fresh call → model acts heavily on transcript signals. Stale call (60+ days ago) → model discounts all transcript signals and drifts toward neutral.
  - **What this suggests**: Call freshness is the on/off switch for the entire transcript model. The model learned on its own that old earnings calls stop being useful.
- `transcript_analyst_qa_sentiment_mean_ffill`
  - **What it measures**: The tone of the analyst Q&A on the earnings call — whether the session was constructive or tense.
  - **What we observe**: Positive Q&A → model calls outperformance. Tense or skeptical Q&A → model calls underperformance. The signal is symmetric and works in both directions.
  - **What this suggests**: When analysts engage positively, the model reads it as external validation. When they push back, it treats it as an early warning — analyst scrutiny tends to precede price weakness.
- `framing_gap_ffill`
  - **What it measures**: Whether management sounds more optimistic on the earnings call than in their formal written filings.
  - **What we observe**: Positive gap → model leans outperformance. No gap or close alignment → little signal. Primarily an upside signal; it does not reliably predict downside.
  - **What this suggests**: When management speaks more confidently than their filings indicate, the model reads it as conviction building before it has been formally communicated to the market.

---

## Model 3 — News

### Observations

- `news_sent_mean_30d`
  - **What it measures**: The average tone of all TD news coverage over the past 30 days. A single headline doesn't move this; it requires a sustained shift.
  - **What we observe**: A month of broadly positive coverage → model calls outperformance. A month of broadly negative coverage → model calls underperformance. Positive sentiment is a stronger outperformance signal than negative sentiment is an underperformance signal.
  - **What this suggests**: Sustained media narrative carries forward into price. Positive is more reliable than negative as a directional predictor — prolonged bad press sometimes overshoots and sets up a recovery rather than further decline.
- `news_count_30d`
  - **What it measures**: How many news articles about TD appeared in the last 30 days, regardless of tone.
  - **What we observe**: High article volume → strong underperformance signal. Quiet coverage → mild outperformance lean. The downside signal is significantly stronger, making this primarily a risk alarm.
  - **What this suggests**: News bursts cluster around bad events. Heavy media attention signals elevated scrutiny and uncertainty — not just activity. Quiet periods offer only mild reassurance.

---

## Per-Fold Deep Dive

The model's 13 evaluation quarters fall into three groups: **5 strong quarters** (65–70% directional accuracy), **2 significantly bad quarters** (below 40%, worse than a coin flip), and 6 moderate quarters near chance. Exact fold dates are in `model_card.md`. Per-fold SHAP breakdowns are in `notebooks/xgb_ensemble_v2_performance.ipynb` (Section 10).

### When the model performs well

Strong quarters share the same condition: TD's price, management tone, and news are all being driven by TD-specific factors rather than sector-wide forces.

- The **price model** leads with 52-week high proximity (`td_dist_52w_high_v2`) in most strong quarters. When price trends are particularly clean, recent return (`td_return_20d`) takes over. Either way, there is a clear directional read on TD's own momentum.
- The **transcript model** is always gated by `days_since_call`. Which secondary signal matters most shifts by period: in the 2023 strong quarters, analyst Q&A tone was the differentiator; from late 2024 onward, AML engagement features moved to the front as the regulatory story dominated earnings calls.
- The **news model** relies on 30-day sentiment (`news_sent_mean_30d`) for outperformance calls and article volume (`news_count_30d`) for underperformance. It works best when coverage is about TD specifically.

The model does not require all three to agree — one model carrying a clear signal while the others are quiet is enough.

### Fold 1 — Mar–Jun 2023 (DA: 0.361) — SVB banking crisis

**What the model was using**: The price model relied on recent momentum (`td_return_20d`) to call outperformance — momentum that was wiped out overnight by the crisis. The transcript model's top signal (`topic_credit_quality_share_ffill`) was correctly pointing to underperformance, but was outvoted. Only the news model's article volume (`news_count_30d`) flagged elevated systemic risk.

**What happened**: Silicon Valley Bank collapsed in March 2023, triggering sector-wide panic. TD and XFN moved together, making the sector-relative return the model is actually predicting effectively random. No signal had a reliable anchor.

**Why the model missed it**: The model had never seen a systemic banking crisis in its training window. During a panic, TD's sector-relative performance is driven by market fear — not by what management said or what the news narrative looked like. The signals were each reading something real; they were just answering the wrong question.

### Fold 8 — Dec 2024–Mar 2025 (DA: 0.388) — AML crisis: priced in, not priced out

**What the model was using**: All three models agreed on underperformance — predicting −1 on 51 of 63 test days. TD was trading 17.7% below its 52-week high, recent momentum was negative, volatility was elevated, executive tone was defensive, and news volume was high. Every signal pointed the same direction.

**What those signals actually reflected**: Every one of these bearish readings was a direct aftershock of the October 2024 AML fine, when TD paid USD 3.09B. During that crisis, TD fell sharply while every peer bank rallied — and the model correctly learned the association: those bearish signals → underperformance. But by December, the fine was fully priced in. TD spent the entire quarter recovering — rising 17% and outperforming XFN by 22 percentage points — while the model kept predicting underperformance based on signals that no longer meant what they had during the crisis.

**Why the model missed it**: The model had never seen the second half of a crisis — only the part where bad signals lead to a falling stock. It had no way to know that a stock 17% below its 52-week high because of a one-time penalty behaves very differently from one 17% below because of an ongoing structural problem. The confusion matrix shows 40 days where the model called −1 and the actual outcome was +1.

---

## Walk-Forward Stability

The transcript and news models are the most consistent. `days_since_call` is the top transcript feature in 10 of 12 fully-evaluated folds — the two exceptions (Folds 1 and 5) are both weak quarters where the transcript model struggled overall. `news_sent_mean_30d` leads the news model's outperformance calls in 10 of 12 folds.

The price model rotates more between folds. `td_dist_52w_high_v2` dominates from 2023 through mid-2024; `td_return_20d` takes over in later folds. Relative-return features (`td_vs_xfn_5d`, `td_vs_xfn_20d`) occasionally lead when sector-relative momentum is the cleaner signal. This rotation is expected — which price feature matters most depends on the market regime that quarter.

---

## Key Investment Takeaways

### When to trust this model's prediction

- **TD is diverging from its peer banks.** A practical check: if RY, BMO, and BNS are all moving in the same direction as TD over the past month, the sector is driving the story and TD-specific signals carry less weight. If TD is moving differently from its peers, the model's inputs are more likely picking up something real about TD specifically.
- **Price momentum is clean and directional.** Strong quarters consistently had TD either near its 52-week high with positive recent returns, or in a sustained drawdown. Choppy, trendless price action weakens the model's strongest signal.

### When to suppress the signal

- **TD just absorbed a major one-time event.** The model cannot tell the difference between a crisis still unfolding and one already priced in. If TD's stock has already fallen sharply due to a known, bounded event, bearish signals may be reading the aftermath — not predicting the future. Fold 8 is the clearest example.
- **A sector-wide shock is happening.** When the whole financial sector is moving together, TD's relative performance is driven by market fear — not by management tone or news narrative. Company-specific signals have no edge in that environment.

