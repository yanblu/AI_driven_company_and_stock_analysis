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

All transcript features are **forward-filled from the earnings call date**. Each feature value updates on the exact trading day the call occurs and holds constant until the next call. `days_since_call` (in trading days) captures this staleness — the model learns how much weight to give transcript features depending on how old the call is.

| Quarter  | Call date  | Features active until | Trading days active |
|----------|------------|-----------------------|---------------------|
| FY2021Q1 | 2021-02-25 | 2021-05-26            | 63                  |
| FY2021Q2 | 2021-05-27 | 2021-08-25            | 63                  |
| FY2021Q3 | 2021-08-26 | 2021-12-01            | 68                  |
| FY2021Q4 | 2021-12-02 | 2022-03-02            | 61                  |
| FY2022Q1 | 2022-03-03 | 2022-05-25            | 58                  |
| FY2022Q2 | 2022-05-26 | 2022-08-24            | 63                  |
| FY2022Q3 | 2022-08-25 | 2022-11-30            | 68                  |
| FY2022Q4 | 2022-12-01 | 2023-03-01            | 61                  |
| FY2023Q1 | 2023-03-02 | 2023-05-25            | 59                  |
| FY2023Q2 | 2023-05-26 | 2023-08-23            | 62                  |
| FY2023Q3 | 2023-08-24 | 2023-11-29            | 68                  |
| FY2023Q4 | 2023-11-30 | 2024-02-28            | 61                  |
| FY2024Q1 | 2024-02-29 | 2024-05-22            | 58                  |
| FY2024Q2 | 2024-05-23 | 2024-08-21            | 63                  |
| FY2024Q3 | 2024-08-22 | 2024-12-04            | 73                  |
| FY2024Q4 | 2024-12-05 | 2025-02-26            | 56                  |
| FY2025Q1 | 2025-02-27 | 2025-05-21            | 58                  |
| FY2025Q2 | 2025-05-22 | 2025-08-27            | 68                  |
| FY2025Q3 | 2025-08-28 | 2025-12-03            | 68                  |
| FY2025Q4 | 2025-12-04 | 2026-04-02            | 82                  |
| FY2026Q1 | 2026-04-03 | 2026-04-09 *(end)*    | 4                   |

*Source: `step1_data_collection/data/raw/transcripts/index.parquet`. "Features active until" = the trading day before the next call (or last day of data for FY2026Q1). FY2025Q4 runs 82 days because TD's Q1 2026 call fell in early April.*

| Feature                                      | Description                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `exec_tone_ffill`                            | Mean CEO + CFO prepared-remarks sentiment                                                      |
| `transcript_analyst_qa_sentiment_mean_ffill` | Average sentiment across all analyst questions and management responses in the Q&A session     |
| `framing_gap_ffill`                          | CEO prepared-remarks tone from the earnings call minus written filing/report tone — whether management sounds more upbeat on the call than on paper |
| `topic_guidance_share_ffill`                 | Share of annotated quarterly passages across transcripts, news, filings, and reports discussing forward guidance |
| `topic_guidance_sentiment_ffill`             | Tone of all-source guidance discussion                                                         |
| `topic_regulatory_AML_share_ffill`           | Share of annotated quarterly passages across transcripts, news, filings, and reports discussing AML and regulatory matters |
| `topic_regulatory_AML_sentiment_ffill`       | Tone of all-source AML/regulatory discussion                                                   |
| `topic_credit_quality_share_ffill`           | Share of annotated quarterly passages across transcripts, news, filings, and reports discussing credit quality |
| `topic_credit_quality_sentiment_ffill`       | Tone of all-source credit quality discussion                                                    |
| `days_since_call`                            | Trading days elapsed since the most recent earnings call                                       |

Topic features cover the three areas most relevant to TD over the evaluation period: forward guidance, AML/regulatory exposure, and credit quality. They are quarter-level, all-source NLP aggregates that are activated from the earnings call date and forward-filled to daily rows. `days_since_call` acts as the staleness signal — the model learns from data how much weight to give the quarterly NLP snapshot depending on how old the call is.

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
