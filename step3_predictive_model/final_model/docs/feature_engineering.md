# Feature Engineering

## Day Convention

Two different day-counting conventions are used across the feature set. It is important not to conflate them.


| Convention        | Definition                                                                    | Used for                                                      |
| ----------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Trading days**  | Market-open days only; weekends, holidays, and exchange closures are excluded | All price return and volatility windows (`5d`, `20d`, `252d`) |
| **Calendar days** | Continuous days including weekends and holidays                               | `days_since_last_news`                                        |


`days_since_call` in the transcript model counts **trading days** — it is derived from the same date index as the price data, where each row is one market-open day.

The distinction matters at short horizons. A 5-trading-day window spans roughly 7 calendar days (one calendar week); a 20-trading-day window spans roughly 28–30 calendar days (one calendar month). The 52-week high window uses 252 trading days, which corresponds to approximately one calendar year of market activity.

---

## Model 1 — Price (6 features)

All return features are **trailing** (past *t−n* to *t*), not forward-looking. All windows are in **trading days**.


| Feature               | Formula                                                                                | Description                                                                                                                                                                                                                                                                                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `td_return_5d`        | `TD.adj_close[t] / TD.adj_close[t−5] − 1`                                              | TD's cumulative price return over the past 5 trading days (≈ 1 calendar week). Short-term momentum signal — captures whether TD has been gaining or losing ground very recently.                                                                                                                                                                                     |
| `td_return_20d`       | `TD.adj_close[t] / TD.adj_close[t−20] − 1`                                             | TD's cumulative price return over the past 20 trading days (≈ 1 calendar month). Medium-term momentum signal — smoother and less noisy than the 5-day window.                                                                                                                                                                                                        |
| `td_vs_xfn_5d`        | `td_return_5d[t] − xfn_return_5d[t]`                                                   | TD's 5-day return minus XFN ETF's 5-day return over the same window. Measures recent sector-relative performance — positive values mean TD has been outpacing the financials sector over the past week.                                                                                                                                                              |
| `td_vs_xfn_20d`       | `td_return_20d[t] − xfn_return_20d[t]`                                                 | TD's 20-day return minus XFN ETF's 20-day return. Measures whether TD has been outpacing or lagging the sector over the past month — a sustained relative advantage or disadvantage.                                                                                                                                                                                 |
| `td_dist_52w_high_v2` | `(TD.adj_close[t] − max(TD.adj_close[t−251..t])) / TD.adj_close[t]`; always ≤ 0        | Distance of TD's current price from its trailing 252-trading-day high (≈ 1 calendar year). A value near zero means TD is trading at or close to its annual peak; a more negative value means TD is well below its yearly high, deeper in drawdown. This captures momentum regime positioning — not just where TD is, but how far it has fallen from its recent best. |
| `td_volatility_20d`   | `std(td_return_1d[t−19..t])`, where `td_return_1d = adj_close[t] / adj_close[t−1] − 1` | Rolling standard deviation of TD's 20 most recent daily returns (20 trading days ≈ 1 calendar month). Measures the turbulence of recent price action — higher values indicate erratic, choppy behaviour; lower values indicate calm, stable trading.                                                                                                                 |


**All price features use `adj_close` (adjusted close)**, which back-adjusts historical prices for dividends and stock splits. Using raw `close` would introduce artificial return gaps on ex-dividend dates, distorting both momentum and volatility signals. TD pays quarterly dividends and XFN pays monthly, so the adjustment is material for both series.

**Design rationale**: short and medium-term momentum (`5d`, `20d`), sector-relative performance at both horizons (`td_vs_xfn`), drawdown positioning relative to the annual range (`td_dist_52w_high_v2`), and volatility regime (`td_volatility_20d`). Volatility scales the ±0.3% classification threshold in practice — high-vol regimes make non-neutral outcomes more frequent, helping the model calibrate its class boundaries.

---

## Model 2 — Transcript + Report (10 features)

All transcript features are **forward-filled** from the most recent earnings call. `days_since_call` is measured in **trading days** — it increments by one for each market-open day elapsed since the call.


| Feature                                      | Description                                                                                                           |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `exec_tone_ffill`                            | Mean CEO + CFO prepared-remarks sentiment (forward-filled)                                                            |
| `transcript_analyst_qa_sentiment_mean_ffill` | Analyst Q&A tone — average sentiment across all analyst questions and management responses                            |
| `framing_gap_ffill`                          | CEO verbal tone minus written filing tone — measures divergence between spoken optimism and formal written disclosure |
| `topic_guidance_share_ffill`                 | Share of the earnings call spent discussing forward guidance                                                          |
| `topic_guidance_sentiment_ffill`             | Tone of guidance discussion                                                                                           |
| `topic_regulatory_AML_share_ffill`           | Share of the call spent discussing AML and regulatory matters                                                         |
| `topic_regulatory_AML_sentiment_ffill`       | Tone of AML/regulatory discussion                                                                                     |
| `topic_credit_quality_share_ffill`           | Share of the call spent discussing credit quality                                                                     |
| `topic_credit_quality_sentiment_ffill`       | Tone of credit quality discussion                                                                                     |
| `days_since_call`                            | Trading days elapsed since the most recent earnings call — the staleness signal for all other transcript features     |


**Design rationale**: executive tone and analyst reception capture sentiment around each earnings call. Topics focus on the three areas most material to TD's risk profile over the evaluation period: forward guidance, AML/regulatory exposure, and credit quality. `days_since_call` acts as the decay signal — the tree learns its own staleness thresholds from data rather than applying a fixed decay function. All features are used undecayed; the model discovers the decay curve implicitly through splits on `days_since_call`.

---

## Model 3 — News (5 features)

News windows are in **calendar days** — articles are stamped with their publication date, and rolling windows count all days including weekends and holidays. `days_since_last_news` is also in **calendar days**.


| Feature                | Description                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| `news_sent_mean_7d`    | Mean LLM sentiment score across all articles published in the last 7 calendar days         |
| `news_sent_mean_30d`   | Mean LLM sentiment score across all articles published in the last 30 calendar days        |
| `news_count_7d`        | Number of articles published in the last 7 calendar days                                   |
| `news_count_30d`       | Number of articles published in the last 30 calendar days                                  |
| `days_since_last_news` | Calendar days elapsed since the most recent article — captures information drought periods |


**Design rationale**: short (7-day) and medium (30-day) windows capture both immediate and sustained shifts in media narrative. Count features measure news volume intensity, which is a signal independent of article tone — elevated coverage tends to cluster around negative catalysts regardless of how individual articles are framed. `days_since_last_news` captures information drought, where the absence of coverage causes the model to lose confidence in news-based signals.