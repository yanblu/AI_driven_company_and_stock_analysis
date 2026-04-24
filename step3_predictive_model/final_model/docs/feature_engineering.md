# Feature Engineering

> **Status: Placeholder** — to be reviewed and completed.

This document covers the rationale and construction details for each feature used across the three sub-models. For the current feature list and source columns, see `model_card.md`.

## Topics to cover

- [ ] Price features: return calculation windows, excess return construction, 52-week distance formula, volatility
- [ ] Transcript features: CEO/CFO sentiment combination, framing gap derivation, topic share and sentiment columns, forward-fill logic and staleness via `days_since_call`
- [ ] News features: rolling sentiment and count windows, `days_since_last_news`
- [ ] Data sources and merge keys (`model_features_daily.parquet`, `nlp_features.parquet`, `TD_TO.parquet`)
- [ ] Any inline engineering steps performed at notebook load time
