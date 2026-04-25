# AI-Driven Company & Stock Analysis — TD Bank

End-to-end pipeline for Toronto-Dominion Bank (`TD.TO`, CAD): collects public data, extracts language signals with an LLM, and trains a predictive model for short-term relative stock performance.

---

## Project layout

```
step1_data_collection/    ← raw data collection, text preprocessing, daily feature table
step2_llm_analysis/       ← zero-shot LLM annotation, sentiment & topic signals, AML narrative
step3_predictive_model/   ← XGBoost ensemble, walk-forward validation, SHAP analysis
```

Each step folder is self-contained: code, data, and docs all live inside it.

---

## Steps

### Step 1 — Data Collection
Collects daily prices (yfinance), macro series (BoC Valet), earnings call transcripts, regulatory filings (Form 40-F), and press releases (TD Newsroom). Preprocesses text into LLM-ready chunks and builds the merged daily feature table.

→ `step1_data_collection/README.md`

### Step 2 — LLM / NLP Analysis
Annotates 3,829 text passages with `gpt-4o-mini` (zero-shot) to extract per-passage sentiment scores and topic labels, aggregated to quarter-level features for Step 3. Includes FinBERT validation and the TD AML signal narrative.

→ `step2_llm_analysis/README.md`

### Step 3 — Predictive Modelling
Three XGBoost models (price, transcript, news) combined via hard majority vote to predict whether TD will outperform, underperform, or match XFN over the next 5 trading days. Validated across 13 walk-forward folds.

| Model | Directional Accuracy |
|---|---|
| Price | 0.530 |
| Transcript | 0.550 |
| News | 0.536 |
| **Hard Majority Vote** | **0.566** |

→ `step3_predictive_model/README.md`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A `.env` file at the project root is required for Step 2 (OpenAI API key). Data collection in Step 1 has no API key requirements — all sources are public.
