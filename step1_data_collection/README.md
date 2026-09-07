# Step 1 — Data Collection

[← Project overview](../README.md) · [Case-study deck](../docs/presentation/summary-slide.pdf) · [Next: LLM analysis →](../step2_llm_analysis/README.md)

Collects the public data used in the TD Bank analysis, prepares financial text for LLM annotation, and builds the daily feature table used by the predictive model.

## At a glance

| | Description |
|---|---|
| **Purpose** | Create a traceable, model-ready dataset from market data and public financial documents |
| **Inputs** | Daily prices, Bank of Canada macro series, earnings calls, regulatory filings, quarterly reports, and TD news releases |
| **Text output** | Cleaned, context-preserving JSONL passages consumed by Step 2 |
| **Model output** | A daily feature table combining market, timing, news-flow, and Step 2 language signals |
| **Provenance** | Source URL, retrieval date, and SHA-256 hash recorded for collected artifacts |

For a conceptual explanation of the sources, preprocessing choices, and token budget, start with the [data-collection methodology](./docs/data_collection_methodology.md). For implementation details, use the structure and execution guide below.

## Folder structure

```
step1_data_collection/
├── src/
│   ├── collectors/
│   │   ├── prices.py           ← daily OHLCV for TD + peers + benchmarks (yfinance)
│   │   ├── macro.py            ← BoC Valet: yields, policy rate, CPI, FX
│   │   ├── transcripts.py      ← earnings-call transcript PDFs from TD IR
│   │   ├── td_ir_reports.py    ← Form 40-F and quarterly Report to Shareholders PDFs
│   │   └── newsroom.py         ← press releases from TD Newsroom
│   ├── preprocess/
│   │   ├── cleaning.py         ← boilerplate strip, whitespace normalisation
│   │   ├── chunking.py         ← semantic chunking (1,800-token target)
│   │   ├── build_chunks.py     ← orchestrates cleaning → chunking → JSONL output
│   │   └── llm_cache.py        ← SHA-256 keyed LLM annotation cache (used by Step 2)
│   ├── features/
│   │   ├── build_daily_features.py  ← merges all sources → model_features_daily.parquet
│   │   └── data_quality_check.py    ← coverage and integrity checks on the feature table
│   └── utils/
│       ├── config.py           ← local path constants and ticker/window settings
│       └── manifest.py         ← provenance CSV helper (auto-called by collectors)
├── data/
│   ├── raw/
│   │   ├── prices/             ← daily OHLCV parquets (one per ticker)
│   │   ├── macro/              ← BoC Valet series (one parquet per series)
│   │   ├── transcripts/        ← PDF + TXT + index.parquet
│   │   ├── td_ir_reports/      ← annual/ (Form 40-F) + quarterly/ (Report to Shareholders)
│   │   └── news/               ← HTML + JSON + index.parquet (TD Newsroom)
│   ├── chunks/                 ← JSONL per source type (LLM-ready) → consumed by Step 2
│   ├── features/
│   │   └── model_features_daily.parquet  ← merged daily feature table → consumed by Step 3
│   └── manifest.csv            ← provenance log (URL, SHA-256, retrieval date per artifact)
└── docs/
    └── data_collection_methodology.md  ← sources, preprocessing pipeline, token budget
```

## Key outputs

`data/features/model_features_daily.parquet` — ~1,285 trading-day rows fed into Step 3:

- **Price / momentum** — rolling returns, volatility, 52-week distance, sector-relative momentum
- **Macro** — BoC policy rate, GoC yield curve, USD/CAD
- **News flow** — 30-day rolling sentiment mean and article count
- **NLP event features** — 11 LLM-scored signals from Step 2, forward-filled from earnings call date
- **Timing** — days since last earnings call, earnings-week flag

`data/chunks/*.jsonl` — 4,491 text chunks across 728 documents → passed to Step 2 for LLM annotation.

## Execution order

All scripts are run from the **project root**. Each step depends on the previous.

```bash
# 1. Collect raw data
python step1_data_collection/src/collectors/prices.py
python step1_data_collection/src/collectors/macro.py
python step1_data_collection/src/collectors/transcripts.py
python step1_data_collection/src/collectors/td_ir_reports.py
python step1_data_collection/src/collectors/newsroom.py

# 2. Build LLM-ready chunks → Step 2 annotates these
python step1_data_collection/src/preprocess/build_chunks.py

# 3. (Run Step 2 — LLM annotation)

# 4. Build the daily feature table (requires Step 2 output)
python step1_data_collection/src/features/build_daily_features.py

# 5. Validate
python step1_data_collection/src/features/data_quality_check.py
```

All collectors are idempotent — re-running skips already-downloaded files and appends to `data/manifest.csv`. After the text chunks are built, continue with [Step 2 — LLM Analysis](../step2_llm_analysis/README.md), then return here to build the final daily feature table.
