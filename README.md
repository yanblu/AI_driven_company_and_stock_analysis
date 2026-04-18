# AI-Driven Analysis of TD Bank (Canadian Market)

End-to-end AI/analytics MVP for Toronto-Dominion Bank (TSX: `TD.TO`, CAD), covering:

1. Data collection from public sources (this repo)
2. LLM-based topic + sentiment analysis of textual data
3. Predictive modeling of TD.TO daily returns
4. Storytelling visualizations for a mixed-audience presentation

## Scope (MVP)

- **Subject**: Toronto-Dominion Bank, Canadian market, CAD pricing only
- **Window**: uniform 5-year lookback (2021-01-01 → present) across all data types
- **Step 2 approach**: zero-shot LLM for topic + sentiment (optional FinBERT sanity-check baseline)

## Repository layout

```
data/
  raw/          # immutable pulls, organized by source
    prices/          # yfinance OHLCV parquets
    macro/           # BoC Valet, StatCan, FRED
    filings/         # SEC EDGAR 40-F and 6-K
    transcripts/     # TD IR quarterly call transcripts (PDF + parsed)
    news/            # TD newsroom press releases + dedup info
  processed/
    chunks/          # LLM-ready text chunks with metadata (JSONL per source)
    llm_annotations/ # cached LLM outputs keyed by content hash
    features/        # joined feature tables for Step 3
  manifest.csv       # provenance for every artifact

src/
  collectors/   # one module per data source
  preprocess/   # cleaning, section/speaker tagging, chunking
  utils/        # config, manifest bookkeeping

notebooks/     # exploratory analysis (Steps 2-4)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.collectors.prices
python -m src.collectors.macro
python -m src.collectors.edgar
python -m src.collectors.transcripts
python -m src.collectors.newsroom
python -m src.preprocess.build_chunks
```

See [`data/README_data.md`](data/README_data.md) for detailed source documentation.
