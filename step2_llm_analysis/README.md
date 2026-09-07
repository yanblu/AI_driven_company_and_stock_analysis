# Step 2 — LLM / NLP Analysis

[← Project overview](../README.md) · [← Step 1: Data collection](../step1_data_collection/README.md) · [Case-study deck](../docs/presentation/summary-slide.pdf) · [Next: Predictive modelling →](../step3_predictive_model/README.md)

Transforms the financial text prepared in Step 1 into structured sentiment and topic signals. Passage-level annotations are aggregated into quarterly features that can be joined to market data in Step 3.

## At a glance

| | Description |
|---|---|
| **Purpose** | Convert unstructured financial language into consistent, model-ready numerical features |
| **Input** | Context-preserving passages from earnings calls, news releases, regulatory filings, and quarterly reports |
| **Method** | Zero-shot structured annotation with `gpt-4o-mini`, cached by content hash |
| **Output** | Quarterly sentiment, topic, and narrative features consumed by the predictive model |
| **Supporting analysis** | FinBERT sign-agreement comparison and an AML narrative case study |

For the quickest analytical review, open [`02_annotation.ipynb`](./notebooks/02_annotation.ipynb). Prompt design, signal definitions, and interpretation are documented in the [LLM/NLP methodology](./docs/llm_nlp_methodology_analysis.md).

## Folder structure

```
step2_llm_analysis/
├── notebooks/
│   └── 02_annotation.ipynb            ← Sentiment/topic time-series, FinBERT comparison, NLP feature summary
├── docs/
│   ├── llm_nlp_methodology_analysis.md ← Annotation method, prompt design, signal definitions, AML narrative
│   └── td-aml-narrative.pdf           ← Exported AML three-phase signal story
├── src/
│   ├── config.py                      ← Local path constants (data lives inside this folder)
│   ├── annotate.py                    ← LLM annotation runner with sha256 cache
│   ├── aggregate_features.py          ← Rolls per-passage annotations up to quarter-level features
│   └── finbert_baseline.py            ← Independent FinBERT sign-agreement comparison
└── data/
    ├── llm_annotations/               ← Per-passage annotation cache (one JSONL per source type)
    └── features/
        ├── nlp_features.parquet       ← Quarter-level NLP feature matrix (22 quarters × 35 columns) → fed to Step 3
        └── finbert_comparison_transcripts.parquet
```

> **Note:** Text chunks (input to annotation) are produced by Step 1 and live in `step1_data_collection/data/chunks/`.

## Key outputs

`data/features/nlp_features.parquet` — 22 quarters × 35 columns fed into Step 3:

- **8 sentiment columns** — mean sentiment score by speaker role and document type (CEO prep, CFO prep, analyst Q&A, news, filings)
- **24 topic columns** — for each of 12 business topics: share of passages tagged, and mean sentiment on those passages
- **1 diversity column** — topic entropy (how concentrated vs. broad the quarter's discussion was)

**3,884 passages annotated** across transcripts, news releases, 40-F filings, and quarterly reports. Analysis scope: FY2021Q1–FY2026Q1 (21 complete quarters; 3,857 passages in analysis — FY2026Q2 excluded as incomplete).

## Execution order

All scripts are run from the **project root**. To create new annotations, copy [`.env.example`](../.env.example) to `.env` and provide `OPENAI_API_KEY`, or set the key in the shell environment. Existing cached annotations can be inspected without making new API calls.

```bash
# 1. Annotate all passages with gpt-4o-mini (reads chunks from step1_data_collection/data/chunks/)
python step2_llm_analysis/src/annotate.py

# 2. Roll per-passage scores up to one row per quarter
python step2_llm_analysis/src/aggregate_features.py

# 3. (Optional) FinBERT validation — compare sign agreement with LLM labels
python step2_llm_analysis/src/finbert_baseline.py

# 4. Explore results in the notebook
# open step2_llm_analysis/notebooks/02_annotation.ipynb
```

Steps 1 and 2 are idempotent — re-running skips already-cached chunks and overwrites the feature parquet in place. The aggregated feature table is consumed by [Step 3 — Predictive Modelling](../step3_predictive_model/README.md).

## Key finding

TD's AML enforcement cycle is visible end-to-end in the language signals. `regulatory_AML` topic share doubled from ~11% (FY2022 baseline) to 30% at the consent-order quarter (FY2024Q4), while AML sentiment fell 86%. CEO prepared-remarks sentiment hit 0.000 — the only zero in the 21-quarter dataset — and fell below the mandatory annual filing for the first time. Recovery signals (framing gap turning positive, guidance share spiking to a five-year high) appeared from FY2025Q1 onward. See the [full methodology and narrative analysis](./docs/llm_nlp_methodology_analysis.md).
