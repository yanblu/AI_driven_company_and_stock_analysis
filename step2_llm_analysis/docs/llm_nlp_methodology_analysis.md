# LLM / NLP Annotation Methodology

This step reads all TD Bank earnings call transcripts, news releases, and regulatory filings collected in Step 1, and uses a large language model to extract two signals from each passage: how positive or negative the language is (sentiment), and what business topic it covers. Those signals are then summarised to one row per quarter and passed to Step 3 as model inputs.

## Contents

| File / Folder | Description |
|---|---|
| `notebooks/02_annotation.ipynb` | Sentiment/topic time-series, FinBERT comparison, NLP feature summary |
| `src/annotate.py` | Runs the LLM annotation; handles caching and retries |
| `src/aggregate_features.py` | Rolls per-passage annotations up to quarter-level features |
| `src/finbert_baseline.py` | Independent FinBERT baseline for validation |
| `data/llm_annotations/*.jsonl` | Raw per-passage annotation cache (one file per source type) |
| `data/features/nlp_features.parquet` | Quarter-level feature matrix (22 quarters × 35 columns) used by Step 3 |

---

## Method

### Why an LLM?

Financial language is dense with jargon. An off-the-shelf model like FinBERT handles some of it, but misses a lot of management intent — a CEO saying "we are making necessary investments in our compliance infrastructure" reads as neutral to a word-counting model, but carries clear negative framing. `gpt-4o-mini` was used at `temperature=0` (fully deterministic) with a fixed prompt. The full run covered 3,829 passages and cost ~$0.70.

### How each passage is labelled

Each passage is sent to the model with this request:

```
Document type: {source_type}  Quarter: {fiscal_quarter}  Section: {section}
Speaker: {speaker_name} ({speaker_role})

Passage: """ {text} """

Respond with exactly this JSON and nothing else:
{ "sentiment": "positive"|"neutral"|"negative",
  "sentiment_score": <float -1.0 to 1.0>,
  "topics": [<1-3 labels>],
  "key_quote": "<one sentence max 20 words>" }
```

Three filters are applied before annotation to remove noise:
1. Passages under 50 tokens (single-line stubs, headers)
2. Passages where more than 40% of tokens are numbers or symbols (raw financial tables)
3. Transcript IR/housekeeping passages that open calls but contain no business content

### Sentiment: label vs. score

The model returns both a category (`positive`, `neutral`, or `negative`) and a numeric score (`-1.0` to `+1.0`) in the same response. Both come from the model's reading of the passage — there is no coded rule like "score above 0.2 = positive." The only constraint applied in code is clamping the score to the `[-1, 1]` range (`src/annotate.py`).

**Only the numeric score is used downstream.** The category label is stored for inspection and debugging but does not feed into any quarter-level features or model inputs.

---

## Data sources

| Source | Passages | Documents |
|---|---:|---|
| Earnings call transcripts | 834 | 20 quarterly calls (FY2021Q2–FY2026Q1) |
| TD Newsroom press releases | 1,019 | 688 releases |
| Annual 40-F filings | 933 | 5 filings (FY2021–FY2025) |
| Quarterly Reports to Shareholders | 1,043 | 16 reports (Q1–Q3, FY2021–FY2026Q1) |
| **Total** | **3,829** | |

Analysis scope: FY2021Q2–FY2026Q1 (20 complete quarters). FY2026Q2 is excluded from charts and analysis (incomplete quarter — news only); records are kept in the feature file for Step 3.

---

## Signals extracted

**Sentiment (8 columns per quarter):**

| Feature | What it captures |
|---|---|
| `transcript_ceo_prep_sentiment_mean` | CEO prepared-remarks tone |
| `transcript_cfo_prep_sentiment_mean` | CFO prepared-remarks tone |
| `transcript_exec_qa_sentiment_mean` | Executive responses during analyst Q&A |
| `transcript_analyst_qa_sentiment_mean` | Analyst question tone in Q&A |
| `transcript_sentiment_std` | How much sentiment varied within the quarter |
| `news_sentiment_mean` | Press release tone |
| `report_40f_sentiment_mean` | Annual 40-F filing tone (Q4 only) |
| `report_quarterly_sentiment_mean` | Quarterly report tone (Q1–Q3 only) |

**Topic features (24 columns):** for each of the 12 topic labels below, two numbers per quarter — the share of passages tagged with that topic, and the average sentiment score on those passages.

**Diversity (1 column):** `topic_entropy` — how spread out the discussion was across topics that quarter. A low score means the quarter's content was concentrated on one or two themes.

---

## Topic taxonomy

12 labels; each passage can receive between 1 and 3:

| Label | What it covers |
|---|---|
| `NIM` | Net interest margin, deposit pricing, spread commentary |
| `credit_quality` | Loan losses, provisions, impairments |
| `capital` | Capital ratios, dividends, share buybacks |
| `US_retail` | TD Bank America segment |
| `Canadian_personal` | Canadian Personal & Commercial Banking |
| `wealth_wholesale` | TD Wealth + TD Securities |
| `regulatory_AML` | AML consent order, compliance, regulatory actions |
| `macro_outlook` | Interest rate outlook, inflation, recession risk |
| `cost_efficiency` | Operating expenses, headcount |
| `guidance` | Forward-looking statements, management targets |
| `M_and_A` | Acquisitions and divestitures (First Horizon, Schwab stake) |
| `other` | Everything else |

---

## What the signals show: the TD AML cycle

> All figures below are from `data/features/nlp_features.parquet`, validated in `notebooks/02_annotation.ipynb`.

TD's AML enforcement cycle left a clear, measurable trail in the language signals — rising before the public announcement and recovering after resolution.

### Phase 1 — Build-up (FY2023Q2 → FY2024Q3)

The share of content discussing regulatory and AML matters climbed steadily while the tone on those passages fell toward neutral. This happened across eight consecutive quarters before the consent order was announced.

| Quarter | AML share | AML sentiment |
|---|---:|---:|
| FY2022Q1 | 11.1% | 0.239 |
| FY2023Q2 | 11.5% | 0.124 |
| FY2023Q3 | 17.6% | 0.070 |
| FY2024Q2 | 20.5% | 0.152 |
| FY2024Q3 | 22.4% | 0.055 |
| FY2024Q4 | 30.0% | 0.033 |

AML share roughly doubled from the FY2022 baseline. AML sentiment fell 86% over the same period.

### Phase 2 — Peak enforcement (FY2024Q4)

CEO prepared-remarks sentiment dropped to 0.000 — the only zero in the 21-quarter dataset — and for the first time, fell below the tone of the mandatory annual filing. The framing gap (CEO prepared remarks minus filing tone) turned negative: management was less optimistic than their own required disclosures.

| Quarter | CEO prep | Filing | Gap |
|---|---:|---:|---:|
| FY2024Q3 | 0.600 | 0.029 | +0.571 |
| FY2024Q4 | **0.000** | 0.137 | **−0.137** |

The FY2024Q4 filing sentiment (0.137) reflects the full fiscal year's narrative — averaged across earlier quarters that were not under enforcement — so it understates the Q4 stress. Despite that, CEO tone still fell below it. AML topic share simultaneously hit its dataset peak at 30% — nearly one in three annotated passages that quarter covered regulatory or compliance matters.

### Phase 3 — Recovery (FY2025Q1 → FY2026Q1)

Two signals marked the recovery: the framing gap returned positive, and guidance discussion surged to a five-year high.

**Framing gap** — CEO sentiment back above filing tone from FY2025Q1 onward, recovering toward the pre-crisis norm:

| Quarter | CEO prep | Filing | Gap |
|---|---:|---:|---:|
| FY2025Q1 | 0.600 | 0.188 | +0.412 |
| FY2025Q3 | 0.700 | 0.269 | +0.456 |
| FY2026Q1 | **0.800** | **0.306** | **+0.494** |

**Guidance share** — how often management made forward-looking commitments:

| Quarter | Guidance share | Guidance sentiment |
|---|---:|---:|
| FY2024Q4 | 8.8% | 0.288 |
| FY2025Q1 | **17.8%** | 0.261 |
| FY2025Q2 | 15.2% | 0.456 |
| FY2026Q1 | 16.5% | **0.545** |

FY2025Q1 had the highest guidance share in the five-year dataset. By FY2026Q1, analyst Q&A sentiment reached its dataset high of **0.529** — buy-side scrutiny had turned constructive.

---

## Validation

**Manual spot-check:** A stratified sample of 30 passages with LLM labels and key quotes is in `notebooks/02_annotation.ipynb` Section 5.

**Event checks:** Known TD events were tested against observed signals:

| Event | Expected signal | Quarter |
|---|---|---|
| AML consent order (Oct 10, 2024) | `regulatory_AML` share spike; CEO sentiment drop | FY2024Q4 |
| BoC rate hiking cycle (Mar 2022) | `macro_outlook` share spike; `NIM` tone improvement | FY2022Q2 |
| First Horizon deal collapse (May 2023) | `M_and_A` share spike; negative sentiment | FY2023Q3 |
| Post-AML resolution | CEO sentiment recovery toward positive | FY2026Q1 |

**FinBERT comparison:** FinBERT (`ProsusAI/finbert`) was run independently on all 856 transcript passages. Sign agreement with the LLM: **86.8%**, Pearson r = 0.501. FinBERT is systematically more cautious — the main divergence is passages where the LLM scores positive and FinBERT scores neutral, which reflects optimistic management framing that a lexical model reads as factually neutral. See `notebooks/02_annotation.ipynb` Section 6.

---

## Limitations

**Model version drift:** Results used `gpt-4o-mini` at `temperature=0`. OpenAI does not guarantee identical outputs across model updates. The `PROMPT_VERSION` constant in `src/annotate.py` versions each annotation pass so future re-runs can be compared. For production use, pin to a dated model snapshot (e.g., `gpt-4o-mini-2024-07-18`).

**Chunk boundary effects:** Chunking can split a passage mid-sentence. A chunk that opens mid-thought may get an inaccurate label because the model lacks surrounding context. Not corrected in this pass to avoid invalidating the existing cache.
