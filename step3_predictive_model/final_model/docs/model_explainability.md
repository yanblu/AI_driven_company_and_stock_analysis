# Model Explainability

SHAP analysis is run on pooled out-of-sample test data across all 13 walk-forward folds. Each fold contributes its own test-set predictions and SHAP values; these are combined before computing mean |SHAP| per feature. Beeswarm and walk-forward importance plots are generated in `notebooks/xgb_ensemble_v2_performance.ipynb` (Sections 8–9).

Results are shown separately for the **+1** (TD outperforms XFN) and **−1** (TD underperforms XFN) classes, since XGBoost produces class-specific SHAP values under `multi:softprob`.

---

## Model 1 — Price

### Mean |SHAP| across all OOS folds

| Rank | Feature | Mean \|SHAP\| (+1) | Mean \|SHAP\| (−1) |
|---|---|---|---|
| 1 | `td_dist_52w_high_v2` | **0.197** | **0.214** |
| 2 | `td_vs_xfn_20d` | 0.123 | 0.091 |
| 3 | `td_return_20d` | 0.122 | 0.159 |
| 4 | `td_vs_xfn_5d` | 0.103 | 0.087 |
| 5 | `td_volatility_20d` | 0.081 | 0.123 |
| 6 | `td_return_5d` | 0.064 | 0.093 |

### Observations

- **`td_dist_52w_high_v2` dominates both classes** — how close or far TD's stock price sits relative to its 52-week high is the single most influential price signal, acting as a regime anchor for the model.
  - **What drives the call**: When TD is trading near its 52-week high, the model reads this as a healthy momentum environment and becomes more likely to call **outperformance**. When TD is well below its 52-week high — sitting in a prolonged drawdown — the model reads this as a weak regime and is more likely to call **underperformance**. Think of it as: being near the top of the year's range means the stock has been doing well and tends to keep doing well vs. the sector.

- **Outperformance is predicted by how TD is doing *relative to the financial sector*, not in absolute terms.**
  - **`td_vs_xfn_20d` / `td_vs_xfn_5d`**: When TD has been outpacing the XFN financials ETF over the past 20 or 5 days, the model sees this as continuing momentum and is more likely to call **outperformance** in the coming week. When TD has been lagging the sector, the model leans toward **underperformance**. Relative strength matters more than raw price gains.
  - **`td_return_20d` / `td_return_5d`**: Strong absolute gains generally push the model toward **outperformance**. However, when recent gains have been large *and* the stock has been unusually volatile, the model can flip toward **underperformance** — treating a sharp, volatile move as overextended rather than sustainable.

- **`td_volatility_20d`** — how much TD's price has been swinging day-to-day over the last month — matters more for predicting underperformance (0.123) than outperformance (0.081).
  - **What drives the call**: When daily price swings have been large and erratic, the model becomes more likely to call **underperformance**. High volatility amplifies the chance of a weak 5-day stretch relative to the sector. When price action is calm and steady, the model sees a quiet, stable environment and is mildly more likely to call **outperformance**.

---

## Model 2 — Transcript

### Mean |SHAP| across all OOS folds

| Rank | Feature | Mean \|SHAP\| (+1) | Mean \|SHAP\| (−1) |
|---|---|---|---|
| 1 | `days_since_call` | **0.227** | **0.260** |
| 2 | `topic_regulatory_AML_share_ffill` | 0.123 | 0.071 |
| 3 | `transcript_analyst_qa_sentiment_mean_ffill` | 0.087 | 0.079 |
| 4 | `framing_gap_ffill` | 0.060 | 0.038 |
| 5 | `topic_credit_quality_share_ffill` | 0.044 | 0.034 |
| 6 | `topic_credit_quality_sentiment_ffill` | 0.040 | 0.078 |
| 7 | `topic_regulatory_AML_sentiment_ffill` | 0.040 | 0.023 |
| 8 | `topic_guidance_sentiment_ffill` | 0.037 | 0.023 |
| 9 | `exec_tone_ffill` | 0.026 | 0.075 |
| 10 | `topic_guidance_share_ffill` | 0.023 | 0.033 |

### Observations

- **`days_since_call`** is the most influential feature in the transcript model (importance 0.227 for +1, 0.260 for −1) — how recently an earnings call happened matters more than what was actually said.
  - **What drives the call**: This feature does not predict a direction on its own — instead it controls how much the model trusts the other transcript signals. In the first few weeks after an earnings call, the model relies heavily on what was discussed and how executives sounded. As the quarter ages and the call becomes stale (past 60 days), the model stops leaning on transcript signals and drifts toward a neutral view — the call's information has been digested by the market. The model learns this decay automatically, without being given an explicit rule.

- **Outperformance calls are driven by *what topics were raised*, not just how they sounded:**
  - **`topic_regulatory_AML_share_ffill`** — when a large portion of the earnings call was spent discussing the AML (anti-money laundering) regulatory situation, the model is more likely to call **outperformance**. This is counterintuitive: the model appears to treat active, substantive engagement with the AML issue as a sign of management working toward resolution, rather than as a red flag. Calls where AML is barely mentioned provide little signal in either direction.
  - **`framing_gap_ffill`** — when management speaks more positively and confidently on the call than their written regulatory filings suggest, the model leans toward **outperformance**. The verbal optimism relative to conservative filing language may signal confidence that hasn't yet been fully communicated in writing. When there is no gap — management sounds equally cautious both verbally and in filings — there is less signal for the upside.

- **Underperformance calls are driven by *how* management and analysts sounded, not just the topics raised:**
  - **`exec_tone_ffill`** — this is the clearest downside signal in the transcript model. When executives sound cautious, muted, or guarded in their prepared remarks, the model is roughly three times more likely to use this as a downside signal than it uses positive executive tone as an upside signal. Put simply: confident executives do not add much predictive value, but cautious or defensive executives reliably flag risk.
  - **`topic_credit_quality_sentiment_ffill`** — when the language used around credit quality sounds negative or deteriorating, the model flags **underperformance**. This is the sharpest negative trigger in the transcript model: not merely that credit quality was mentioned, but that it was discussed in a worrying way.
  - **`transcript_analyst_qa_sentiment_mean_ffill`** — the overall tone of the analyst question-and-answer session drives both directions equally. When analysts and management interact positively and constructively, the model leans toward **outperformance**; when the Q&A is tense or negative, it leans toward **underperformance**. This is the most balanced two-way signal in the transcript model.

- **Asymmetry summary**: the model predicts upside from *what is discussed* (AML engagement, framing divergence) and downside from *how management sounds* (executive tone, credit quality language). A call that raises difficult topics in a calm, confident tone is much less likely to trigger an underperformance call than one where tone itself signals distress.

---

## Model 3 — News

### Mean |SHAP| across all OOS folds

| Rank | Feature | Mean \|SHAP\| (+1) | Mean \|SHAP\| (−1) |
|---|---|---|---|
| 1 | `news_sent_mean_30d` | **0.163** | 0.103 |
| 2 | `news_count_7d` | 0.064 | 0.098 |
| 3 | `news_sent_mean_7d` | 0.061 | 0.061 |
| 4 | `news_count_30d` | 0.052 | **0.114** |
| 5 | `days_since_last_news` | 0.027 | 0.036 |

### Observations

- **`news_sent_mean_30d`** — the average tone of news coverage over the past 30 days — is the strongest outperformance signal (0.163) and the second-strongest underperformance signal (0.103).
  - **What drives the call**: When TD has been receiving consistently positive press coverage over the past month, the model is more likely to call **outperformance** — a durable positive narrative tends to carry forward into near-term price performance. When sentiment has been persistently negative, the primary call is **underperformance**. However, at very extreme levels of negativity, the model can occasionally flip toward **outperformance** — reflecting a pattern where the market over-reacts to bad news and the stock becomes poised for a recovery. A single good or bad headline does not trigger this; it requires a sustained shift in media tone.

- **`news_count_30d`** — the total number of news articles in the past 30 days — is the strongest underperformance signal (0.114) and only a minor outperformance signal (0.052).
  - **What drives the call**: A high volume of press coverage over the past month is a risk signal, *regardless of whether the articles are positive or negative*. News bursts tend to cluster around negative catalysts — regulatory developments, analyst downgrades, earnings concerns, sector stress — so more articles means more scrutiny and more uncertainty. When coverage is quiet, the model sees less risk and mildly favours **outperformance**.

- **`news_count_7d`** — recent news volume in the past week — shows the same pattern but at shorter range.
  - **What drives the call**: A spike in articles in the last seven days is an even more acute **underperformance** signal (0.098) than it is for outperformance (0.064). A recent news burst is an earlier warning of imminent risk than a slow 30-day build.

- **`news_sent_mean_7d`** — short-term news tone — is equally balanced across both directions (0.061 / 0.061).
  - **What drives the call**: A week of positive coverage nudges the model toward **outperformance**; a week of negative coverage nudges toward **underperformance**. Because it is so balanced, this is a two-way signal rather than a lopsided risk indicator.

- **`days_since_last_news`** is consistently the weakest feature for both calls (0.027 / 0.036).
  - **What drives the call**: When there has been no news for an unusually long time, the model loses confidence in the news signal entirely and reverts toward a neutral view. An information drought adds almost no directional value compared to actual content and volume.

---

## Per-Fold Deep Dive

The model is evaluated on 13 consecutive 3-month (quarterly) test windows covering **March 2023 – April 2026**. Directional accuracy measures what share of active trading signals pointed in the right direction (random chance = 50%).

| Fold | Period | Directional Accuracy | Assessment |
|---|---|---|---|
| 1 | Mar–Jun 2023 | 0.361 | Significantly below chance |
| 2 | Jun–Sep 2023 | **0.694** | Strong |
| 3 | Sep–Dec 2023 | 0.538 | Slightly above chance |
| 4 | Dec 2023–Mar 2024 | **0.657** | Strong |
| 5 | Mar–Jun 2024 | 0.572 | Moderate |
| 6 | Jun–Sep 2024 | 0.594 | Moderate |
| 7 | Sep–Dec 2024 | **0.686** | Strong |
| 8 | Dec 2024–Mar 2025 | 0.388 | Worst fold — significantly below chance |
| 9 | Mar–Jun 2025 | 0.459 | Below chance |
| 10 | Jun–Sep 2025 | **0.663** | Strong |
| 11 | Sep–Dec 2025 | 0.457 | Just below chance |
| 12 | Dec 2025–Mar 2026 | **0.691** | Strong |
| 13 | Mar–Apr 2026 | 0.600 | Moderate (small sample, 20 days) |

### Fold 1 — Mar–Jun 2023 (DA: 0.361)

The model's weakest early quarter. This period followed the collapse of Silicon Valley Bank and Signature Bank in March 2023, which triggered a global wave of concern about regional bank and large bank stability. TD's stock moved in ways heavily driven by sector-wide fear rather than TD-specific fundamentals. The transcript model's signals from TD's own earnings call — including early AML-related language — were drowned out by macro contagion the model had never seen in training data. The model confidently called directions that the market reversed, producing below-chance accuracy.

### Fold 2 — Jun–Sep 2023 (DA: 0.694)

The model's joint-best quarter. Banking stress fears receded and the market re-focused on company fundamentals. TD's stock began trading more on its own news flow and earnings call signals. The AML topic became a more established part of the investment narrative, meaning the transcript model's AML-related features had consistent and reliable signal. News volume and sentiment were also more interpretable — coverage was material but not driven by unpredictable external shocks. The model's three signal sources aligned well.

### Fold 3 — Sep–Dec 2023 (DA: 0.538)

Near-random accuracy. This quarter saw interest rate expectations shift significantly as the Bank of Canada and the Fed reached their terminal rates. TD, like other Canadian banks, was heavily influenced by rate-path uncertainty — a macro factor not well-captured by the model's price, transcript, or news features. The model produced slight but unreliable signal.

### Fold 4 — Dec 2023–Mar 2024 (DA: 0.657)

A strong quarter. The macro rate environment stabilised after the rate hiking cycle ended, and investor focus shifted to TD-specific fundamentals: AML resolution progress and earnings quality. The transcript and news models both had clean, consistent signals — management tone was interpretable, and news sentiment showed durable trends rather than noise spikes.

### Folds 5–6 — Mar–Sep 2024 (DA: 0.572, 0.594)

Moderate, consistently above chance. These two quarters were characterised by an AI-driven broad market rally where growth stocks led. Canadian financials including TD outperformed modestly but lagged the broader market narrative. The model produced useful but not outstanding signal — TD-specific features had signal, but sector-relative moves were partially tied to macro themes the model does not directly capture.

### Fold 7 — Sep–Dec 2024 (DA: 0.686)

One of the model's strongest quarters, coinciding with TD Bank's AML settlement announcement in October 2024 (a US$3.1 billion penalty). This was a high-information event that the model's transcript and news features were directly built to capture: AML topic share rose sharply, management tone shifted in a measurable way once the settlement path became clearer, and news volume spiked then normalised. The model's AML-related signals translated cleanly into correct directional calls in the period surrounding the announcement.

### Fold 8 — Dec 2024–Mar 2025 (DA: 0.388)

The worst quarter in the entire evaluation period, and the clearest example of model failure. This window covers the onset of US tariff policy escalation in early 2025, which produced sharp, correlated declines across Canadian bank stocks and the broader market. TD's stock movements during this period were driven almost entirely by macro and political news that is not captured anywhere in the model's feature set — not by price momentum signals, not by earnings call language, and not by TD-specific news flow. All three models made confident but wrong directional calls simultaneously. SHAP analysis for this fold shows unusually diffuse feature importance: no single feature dominated, indicating the model was uncertain and effectively guessing in a regime it had not encountered.

### Fold 9 — Mar–Jun 2025 (DA: 0.459)

Slightly below chance, as tariff-related volatility continued to weigh on global markets through the first half of 2025. The model partially recovered from fold 8's level of disruption but remained in noise territory. TD-specific signals (transcript and news) were present and consistent, but were insufficient to overcome macro-driven price moves.

### Fold 10 — Jun–Sep 2025 (DA: 0.663)

A strong recovery. As tariff fears eased and equity markets stabilised, TD-specific signals regained their predictive edge. Price momentum, sector-relative performance, and news sentiment all moved in interpretable, consistent ways. This fold demonstrates that the model's underperformance in folds 8–9 was regime-specific rather than structural.

### Fold 11 — Sep–Dec 2025 (DA: 0.457)

Near-random again. Renewed macro uncertainty, including a second wave of trade policy volatility, temporarily disrupted the model's signal clarity. The three models disagreed with each other — hard majority vote produced a weak signal because the individual models diverged.

### Fold 12 — Dec 2025–Mar 2026 (DA: 0.691)

The model's joint-strongest quarter, alongside fold 2. Conditions closely mirrored fold 2: macro stability had returned, TD was being evaluated on its own merits rather than sector-wide fear, and the transcript and news signals were clean and consistent. Both the news sentiment trend and the transcript tone features aligned strongly with actual price outcomes.

### Fold 13 — Mar–Apr 2026 (DA: 0.600)

A partial quarter (only 20 trading days available at evaluation time). Above chance, but the small sample size limits confidence. No major TD-specific events during this window; the model produced moderate, stable signals.

---

### What the fold pattern tells us

The model performs well — often at 65–70% directional accuracy — in **TD-specific, low-macro-noise environments**: quarters where earnings call signals are fresh, news sentiment has a clear trend, and price action is driven by fundamentals rather than external shocks. It struggles — sometimes falling below random — in **macro-dominated regimes** where the market is moved by forces entirely outside the feature set (banking system stress, tariff escalation, rate shock). This is not a flaw unique to this model; it is a structural limitation of any model that relies on company-specific signals.

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
