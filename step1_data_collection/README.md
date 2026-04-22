# Step 1 — Data Collection

Data collection and preprocessing for the TD Bank AI analysis project.

## Data Location
All raw and processed data lives in the project-root `data/` folder.

```
data/
├── raw/
│   ├── prices/         # Daily OHLCV price data (TD.TO, XFN.TO, TSX, benchmarks)
│   ├── transcripts/    # Earnings call transcripts (JSON)
│   ├── news/           # News articles and press releases
│   └── macro/          # Macro data: yields, CPI, FX rates, VIX
└── processed/
    └── features/
        └── model_features_daily.parquet  # Final merged daily feature table
```

## Source Code
Data collection scripts live in `src/` at the project root:

| Module | Purpose |
|---|---|
| `src/collectors/prices.py` | Historical price data (TD, benchmarks, VIX, Gold, FX) |
| `src/collectors/transcripts.py` | Earnings call transcripts |
| `src/collectors/newsroom.py` | News articles and press releases |
| `src/collectors/macro.py` | Macro data (yields, CPI, rate overnight) |
| `src/collectors/td_ir_reports.py` | TD investor relations reports |
| `src/preprocess/` | Cleaning, chunking, LLM annotation |
| `src/features/build_daily_features.py` | Feature engineering pipeline |

## Data Sources
- **Prices**: Yahoo Finance / direct exchange feeds
- **Transcripts**: Publicly available earnings call transcripts
- **News**: Public news aggregators
- **Macro**: Bank of Canada, Statistics Canada, FRED
