# Feature Engineering

## Day Convention

| Convention        | Definition                                                | Used for                                              |
| ----------------- | --------------------------------------------------------- | ----------------------------------------------------- |
| **Trading days**  | Market-open days only; excludes weekends and holidays     | All price return and volatility windows (`5d`, `20d`, `252d`) |
| **Calendar days** | All days including weekends and holidays                  | News windows and `days_since_last_news`               |

`days_since_call` counts **trading days** — one per market-open day. A 5-trading-day window ≈ 1 calendar week; a 20-trading-day window ≈ 1 calendar month; 252 trading days ≈ 1 calendar year.

---

## Model 1 — Price (6 features)

All return features are **trailing** (past to present, not forward-looking). All windows are in **trading days**. All features use `adj_close` (adjusted close), which corrects for dividends and stock splits — TD pays quarterly dividends and XFN pays monthly, so the adjustment is material.

| Feature               | Formula                                                                                   | Description                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `td_return_5d`        | `TD.adj_close[t] / TD.adj_close[t−5] − 1`                                                | TD's price return over the past 5 trading days (~1 week). Short-term momentum.    |
| `td_return_20d`       | `TD.adj_close[t] / TD.adj_close[t−20] − 1`                                               | TD's price return over the past 20 trading days (~1 month). Medium-term momentum. |
| `td_vs_xfn_5d`        | `td_return_5d[t] − xfn_return_5d[t]`                                                     | TD's 5-day return minus XFN's 5-day return. Whether TD outpaced the sector recently. |
| `td_vs_xfn_20d`       | `td_return_20d[t] − xfn_return_20d[t]`                                                   | TD's 20-day return minus XFN's 20-day return. Sector-relative trend over the past month. |
| `td_dist_52w_high_v2` | `(TD.adj_close[t] − max(TD.adj_close[t−251..t])) / TD.adj_close[t]`; always ≤ 0         | How far TD is from its 52-week high. Near zero = near its annual peak; more negative = deeper drawdown. |
| `td_volatility_20d`   | `std(daily_returns[t−19..t])`, where `daily_return = adj_close[t] / adj_close[t−1] − 1` | Rolling standard deviation of TD's 20 most recent daily returns. Higher = more erratic price action. |

---

## Model 2 — Transcript (10 features)

All transcript features are **forward-filled** from the most recent earnings call. `days_since_call` is in **trading days**.

| Feature                                      | Description                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `exec_tone_ffill`                            | Mean CEO + CFO prepared-remarks sentiment                                                      |
| `transcript_analyst_qa_sentiment_mean_ffill` | Average sentiment across all analyst questions and management responses in the Q&A session     |
| `framing_gap_ffill`                          | CEO verbal tone minus written filing tone — whether management sounds more upbeat on the call than on paper |
| `topic_guidance_share_ffill`                 | Share of the earnings call spent discussing forward guidance                                   |
| `topic_guidance_sentiment_ffill`             | Tone of guidance discussion                                                                    |
| `topic_regulatory_AML_share_ffill`           | Share of the call spent discussing AML and regulatory matters                                  |
| `topic_regulatory_AML_sentiment_ffill`       | Tone of AML/regulatory discussion                                                              |
| `topic_credit_quality_share_ffill`           | Share of the call spent discussing credit quality                                              |
| `topic_credit_quality_sentiment_ffill`       | Tone of credit quality discussion                                                              |
| `days_since_call`                            | Trading days elapsed since the most recent earnings call                                       |

Topics cover the three areas most relevant to TD over the evaluation period: forward guidance, AML/regulatory exposure, and credit quality. `days_since_call` acts as the staleness signal — the model learns from data how much weight to give transcript features depending on how old the call is.

---

## Model 3 — News (5 features)

News windows are in **calendar days** — articles are stamped by publication date, and rolling windows include all days.

| Feature                | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| `news_sent_mean_7d`    | Mean sentiment across articles published in the last 7 calendar days        |
| `news_sent_mean_30d`   | Mean sentiment across articles published in the last 30 calendar days       |
| `news_count_7d`        | Number of articles published in the last 7 calendar days                    |
| `news_count_30d`       | Number of articles published in the last 30 calendar days                   |
| `days_since_last_news` | Calendar days since the most recent article                                 |

Short (7-day) and medium (30-day) windows capture both immediate and sustained shifts in media tone. Article count is a signal independent of tone — elevated coverage tends to cluster around negative events. `days_since_last_news` flags periods with no recent coverage.
