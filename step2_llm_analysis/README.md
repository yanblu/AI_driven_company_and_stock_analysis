# Step 2 — LLM Company Analysis

AI-driven company analysis using earnings call transcripts, news, and filings for TD Bank.
Zero-shot LLM annotation extracts per-chunk sentiment scores and topic labels, aggregated
to quarter-level features for Step 3 predictive modelling.

## Contents

| File / Folder | Description |
|---|---|
| `02_annotation.ipynb` | Main analysis notebook: sentiment/topic time-series, FinBERT comparison, NLP feature summary |
| `step2_holistic_analysis.md` | AML narrative: three-phase LLM-signal story (build-up → enforcement → recovery) |
| `src/annotate.py` | LLM annotation runner with sha256 cache |
| `src/aggregate_features.py` | Aggregates per-chunk annotations to quarter-level features |
| `src/finbert_baseline.py` | FinBERT validation baseline with LLM comparison |
| `data/processed/llm_annotations/*.jsonl` | Per-chunk annotation cache (one file per source type) |
| `data/processed/features/nlp_features.parquet` | Quarter-level NLP feature matrix (22 quarters × 35 columns) for Step 3 |
| `data/processed/features/finbert_comparison_transcripts.parquet` | FinBERT vs LLM comparison data |

---

## Overview

Step 2 applies zero-shot LLM annotation to all 3,829 text chunks collected in Step 1 (after filtering short stubs, table-heavy passages, and administrative intro chunks), extracting two structured signals per chunk:

- **Sentiment score** — a continuous float from −1.0 (very negative) to +1.0 (very positive), plus a label (`positive`, `neutral`, `negative`)
- **Topic labels** — one to three labels from a fixed 12-label taxonomy describing what the passage is about

These per-chunk signals are aggregated to **quarter-level features** (`nlp_features.parquet`) that feed directly into Step 3.

---

## Quarter-level features produced

Output: `data/processed/features/nlp_features.parquet` — 22 quarters × 35 columns.

**Sentiment features (8 columns):**

| Feature | Construction |
|---|---|
| `transcript_ceo_prep_sentiment_mean` | CEO prepared-remarks mean sentiment |
| `transcript_cfo_prep_sentiment_mean` | CFO prepared-remarks mean sentiment |
| `transcript_exec_qa_sentiment_mean` | Executive Q&A responses (CEO + CFO + other_exec) |
| `transcript_analyst_qa_sentiment_mean` | Analyst questions in Q&A |
| `transcript_sentiment_std` | Within-quarter sentiment volatility |
| `news_sentiment_mean` | Newsroom press-release mean sentiment |
| `report_40f_sentiment_mean` | Annual 40-F narrative (Q4 quarters only) |
| `report_quarterly_sentiment_mean` | Quarterly Report to Shareholders (Q1–Q3 only) |

**Topic features (24 columns):** for each of 12 topic labels —
- `topic_{label}_share` — fraction of quarter's chunks tagged with that label
- `topic_{label}_sentiment` — mean sentiment score for chunks tagged with that label

**Diversity feature (1 column):**
- `topic_entropy` — Shannon entropy of topic distribution (excluding `other`); low = concentrated narrative, high = broad discussion

---

## Methods

### Why zero-shot LLM?

An LLM (gpt-4o-mini) can understand financial jargon without fine-tuning, assign multi-label topic tags in a single prompt, and extract a key quote alongside each label. Cost was ~$0.70 for the full run; non-determinism is fixed by `temperature=0` and caching by `sha256(text + prompt + model)`.

### Prompt

**System:** *You are a financial analyst specialising in Canadian banks. Analyse the following passage from a TD Bank document and respond with a JSON object. Be concise and precise. Respond ONLY with the JSON — no prose, no markdown fences.*

**User (per chunk):**
```
Document type: {source_type}  Quarter: {fiscal_quarter}  Section: {section}
Speaker: {speaker_name} ({speaker_role})   [omitted for non-transcript sources]

Passage: """ {text} """

Respond with exactly this JSON and nothing else:
{ "sentiment": "positive"|"neutral"|"negative",
  "sentiment_score": <float -1.0 to 1.0>,
  "topics": [<1-3 labels>],
  "key_quote": "<one sentence max 20 words>" }
```

**Config:** `model=gpt-4o-mini`, `temperature=0`, `BATCH_SIZE=20`, `MAX_RETRIES=2`  
**Result:** 3,829 annotated, 0 failures

---

## Data sources annotated

| File | Chunks | Source documents |
|---|---:|---|
| `transcripts.jsonl` | 834 | 20 quarterly earnings-call transcripts (FY2021Q2–FY2026Q1) |
| `news.jsonl` | 1,019 | 688 TD Newsroom press releases |
| `reports_40f.jsonl` | 933 | 5 annual 40-F filings (FY2021–FY2025) |
| `reports_quarterly.jsonl` | 1,043 | 16 quarterly Reports to Shareholders (FY2021Q1–FY2026Q1 × Q1–Q3) |
| **Total** | **3,829** | |

Three progressive filters were applied before annotation:
1. **Short-token filter** (`< 50 tokens`): excluded name-listing stubs and single-line headers.
2. **Table-heavy / numeric filter** (`> 40% numeric tokens`): excluded 195 raw financial table chunks dominated by numbers with no accompanying narrative.
3. **Admin-intro / neutral-opening filter** (transcripts only): excluded 22 IR-housekeeping and participant-introduction chunks that consistently score 0.000.

**FY2026Q2 analysis exclusion:** FY2026Q2 contains news articles only (incomplete quarter). All notebook charts exclude FY2026Q2; records are retained in `model_features_daily.parquet` for Step 3. Analysis scope: FY2021Q2 – FY2026Q1 (20 complete quarters).

---

## Topic taxonomy

12 labels, non-exclusive (a chunk can receive 1–3 labels):

| Label | What it covers |
|---|---|
| `NIM` | Net interest margin, spread commentary, deposit pricing |
| `credit_quality` | PCL, provisions for credit losses, loan impairments, write-offs |
| `capital` | CET1 ratio, TLAC, dividend declarations, share buybacks |
| `US_retail` | TD Bank America's Most Convenient Bank segment results |
| `Canadian_personal` | Canadian Personal & Commercial Banking segment |
| `wealth_wholesale` | TD Wealth Management + TD Securities / Wholesale Banking |
| `regulatory_AML` | Compliance, AML consent order, regulatory actions, fines |
| `macro_outlook` | BoC/Fed rate commentary, inflation, recession risk, yield curve |
| `cost_efficiency` | Operating expense ratio, operating leverage, headcount |
| `guidance` | Forward-looking statements, management targets, outlook language |
| `M_and_A` | Acquisitions, divestitures (First Horizon deal, Schwab stake sale) |
| `other` | Everything that does not fit the above labels |

---

## Evaluation

**Manual spot-check (30 chunks):** A stratified sample is shown in `02_annotation.ipynb` Section 5 alongside each LLM label and key quote.

**Event-based validation:** Known TD events are tested against observed signals:

| Event | Expected signal | Quarter |
|---|---|---|
| AML consent order (Oct 10, 2024) | `regulatory_AML` share spike; CEO sentiment drop | FY2024Q4 |
| Rate hiking cycle (BoC Mar 2022) | `macro_outlook` share spike; `NIM` tone improvement | FY2022Q2 |
| First Horizon collapse (May 2023) | `M_and_A` share spike; negative sentiment | FY2023Q3 |
| Post-AML resolution | CEO sentiment recovery toward positive | FY2026Q1 |

**FinBERT validation:** FinBERT (`ProsusAI/finbert`) was run on all 856 transcript chunks as an independent domain-specific baseline. Sign agreement: **86.8%** (above the ≥85% strong threshold), Pearson r = 0.501. FinBERT is systematically more cautious than the LLM — the primary divergence is LLM=positive / FinBERT=neutral (240 chunks), reflecting optimistic management framing that FinBERT's news-trained vocabulary reads as factually neutral. This is expected and confirms the LLM is capturing management intent that a lexical model misses. See `02_annotation.ipynb` Section 6 for full confusion matrix, by-role correlations, and divergent chunk examples.

---

## Limitations

**Non-determinism across model versions:** Results were produced with `gpt-4o-mini` at `temperature=0`. OpenAI does not guarantee identical outputs if underlying model weights are updated. The `PROMPT_VERSION` constant in `src/annotate.py` allows re-annotation passes to be versioned and compared; for production use, pin to a specific dated model snapshot (e.g., `gpt-4o-mini-2024-07-18`).

**Chunk boundary effects on sentiment:** Chunking can cut a passage mid-thought. A chunk that opens with hedged language (the tail of a prior thought) may receive an inaccurate label because the LLM lacks surrounding context. Adding a short overlap (1–2 sentences from the preceding chunk) to the prompt would reduce this — not applied in this pass to avoid invalidating the existing annotation cache.
