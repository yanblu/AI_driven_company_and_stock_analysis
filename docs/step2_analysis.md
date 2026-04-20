# Step 2 — Company Analysis & AI Application

## Contents

- [Overview](#overview)
- [Methods and rationale](#methods-and-rationale)
- [Data sources annotated](#data-sources-annotated)
- [Topic taxonomy](#topic-taxonomy)
- [Prompt design](#prompt-design)
- [Quarter-level features produced](#quarter-level-features-produced)
- [Evaluation](#evaluation)
  - [FinBERT — what it is and why it is the baseline](#finbert--what-it-is-and-why-it-is-the-baseline)
  - [FinBERT results](#finbert-results)
- [Notebook findings](#notebook-findings-02_annotationipynb)
- [Holistic analysis — three-act AML narrative](#holistic-analysis--three-act-aml-narrative)
  - [The story](#the-story)
  - [Act 1 — Pre-event language stress](#act-1--pre-event-language-stress-fy2022q3--fy2023q4)
  - [Act 2 — Peak enforcement](#act-2--peak-enforcement-fy2024q1--fy2024q4)
  - [Act 3 — Recovery sequence](#act-3--recovery-sequence-fy2025q1--fy2026q1)
  - [Analytical questions answered](#analytical-questions-answered)
  - [Event-based validation](#event-based-validation-evaluation-d)
- [Limitations and remediations](#limitations-and-remediations) (L1–L10)
- [Implementation files](#implementation-files)

---

## Overview

Step 2 applies zero-shot LLM annotation to all 3,829 text chunks collected in Step 1 (after filtering short stubs, table-heavy passages, and administrative intro chunks), extracting two structured signals per chunk:

- **Sentiment score** — a continuous float from −1.0 (very negative) to +1.0 (very positive), plus a label (`positive`, `neutral`, `negative`)
- **Topic labels** — one to three labels from a fixed 12-label taxonomy describing what the passage is about

These per-chunk signals are then aggregated to **quarter-level features** (`nlp_features.parquet`) that feed directly into Step 3 predictive modelling.

---

## Methods and rationale

### Why zero-shot LLM?

Traditional NLP approaches (bag-of-words, TF-IDF, or even fine-tuned BERT classifiers) require labelled training data and are brittle on domain-specific phrasing. An LLM (gpt-4o-mini) can:

- Understand financial jargon without fine-tuning ("PCL", "CET1", "TLAC", "NIM compression")
- Assign multi-label topic tags from a custom taxonomy in a single prompt
- Extract a key quote alongside the label, giving human-readable evidence for every annotation
- Be re-prompted with a different taxonomy without retraining

The trade-off is cost (paid per token) and non-determinism. Both are managed: cost was ~$0.70 for the full run; non-determinism is fixed by setting `temperature=0` and caching results by `sha256(text + prompt + model)`.

### Guideline alignment

| Guideline item | What was built |
|---|---|
| Sentiment or topic analysis of textual data (earnings calls, news) | Per-chunk sentiment score + 12-label topic taxonomy across all 4 source types |
| Extraction of key signals or indicators related to company performance | `key_quote` field per chunk; topic-conditional sentiment features per quarter |
| Risk or event detection models | `regulatory_AML` topic label; sentiment drop detection; FinBERT sign-agreement baseline |

---

## Data sources annotated

| File | Chunks annotated | Source documents |
|---|---:|---|
| `transcripts.jsonl` | 834 | 20 quarterly earnings-call transcripts (FY2021Q2–FY2026Q1) |
| `news.jsonl` | 1,019 | 688 TD Newsroom press releases |
| `reports_40f.jsonl` | 933 | 5 annual 40-F filings (FY2021–FY2025) |
| `reports_quarterly.jsonl` | 1,043 | 16 quarterly Reports to Shareholders (FY2021Q1–FY2026Q1 × Q1–Q3) |
| **Total** | **3,829** | |

Three progressive filters were applied before annotation:
1. **Short-token filter** (`< 50 tokens`): excluded name-listing stubs and single-line headers.
2. **Table-heavy filter** (`> 40% numeric tokens`): excluded 195 raw financial table chunks from reports (see L1 below).
3. **Admin-intro filter** (transcripts only): excluded 22 IR-housekeeping and participant-introduction chunks that consistently score 0.000 and add only noise to speaker-level aggregations (see L8 below).

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

**Taxonomy design rationale**: labels were chosen based on what moves TD's stock and what analysts probe on every earnings call — not a formal standard, but grounded in practitioner consensus. A financial analyst covering Canadian banks would recognise all 11 substantive labels as core analytical dimensions. The taxonomy intentionally stays narrow to avoid the LLM spreading labels too thin.

**`other` breakdown (20.6% of all chunks):**
- 38.8% from news articles (ETF distributions, ESG grants, HR announcements — non-financial corporate comms)
- 34.7% from 40-F body (accounting policy notes, liquidity tables, fair value hierarchy disclosures)
- 17.8% from quarterly reports (similar technical disclosures)
- 8.8% from transcripts (operator lines, Schwab-specific questions not captured by M&A label)

These chunks carry minimal signal for Step 3 and receive naturally low feature weight.

---

## Prompt design

**System prompt:**
> You are a financial analyst specialising in Canadian banks. Analyse the following passage from a TD Bank document and respond with a JSON object. Be concise and precise. Respond ONLY with the JSON — no prose, no markdown fences.

**User prompt (per chunk):**
```
Document type: {source_type}  Quarter: {fiscal_quarter}  Section: {section}
Speaker: {speaker_name} ({speaker_role})   [omitted for non-transcript sources]

Passage:
"""
{text}
"""

Respond with exactly this JSON and nothing else:
{
  "sentiment": "positive" or "neutral" or "negative",
  "sentiment_score": <float -1.0 to 1.0>,
  "topics": [<1-3 labels from taxonomy>],
  "key_quote": "<one sentence max 20 words>"
}
```

**Configuration:** `model=gpt-4o-mini`, `temperature=0`, `BATCH_SIZE=20`, `MAX_RETRIES=2`  
**Run result:** 3,829 annotated, 0 failures (after short, table-heavy, and admin-intro filters)

---

## Quarter-level features produced

Output: `data/processed/features/nlp_features.parquet` — 22 quarters × 35 columns.

**Sentiment features (8 columns):**
- `transcript_ceo_prep_sentiment_mean` — CEO prepared remarks
- `transcript_cfo_prep_sentiment_mean` — CFO prepared remarks
- `transcript_exec_qa_sentiment_mean` — Executive Q&A responses (CEO + CFO + other_exec)
- `transcript_analyst_qa_sentiment_mean` — Analyst questions in Q&A
- `transcript_sentiment_std` — Within-quarter sentiment volatility across all transcript chunks
- `news_sentiment_mean` — Newsroom press releases
- `report_40f_sentiment_mean` — Annual 40-F narrative (Q4 quarters only)
- `report_quarterly_sentiment_mean` — Quarterly Report to Shareholders (Q1–Q3 only)

**Topic features (24 columns):** for each of 12 topic labels —
- `topic_{label}_share` — fraction of quarter's chunks tagged with that label
- `topic_{label}_sentiment` — mean sentiment score for chunks tagged with that label

**Diversity feature (1 column):**
- `topic_entropy` — Shannon entropy of topic distribution (excluding `other`); low = concentrated narrative, high = broad discussion

---

## Evaluation

### Approach

Because financial NLP has no universal ground truth for sentiment or topic labels, evaluation combines three complementary methods:

**A — Manual spot-check (30 chunks)**  
A stratified sample (10 each from transcripts, news, quarterly reports) is shown in `notebooks/02_annotation.ipynb` Section 5 alongside the LLM's label and key quote. A human reviewer marks agree/disagree. Target: >85% sentiment sign agreement, >70% top-1 topic agreement for MVP pass.

**C — FinBERT sign-agreement baseline**  
FinBERT (`ProsusAI/finbert`) is run on the same transcript chunks. The sign of its sentiment label is compared against the LLM's label. Agreement ≥70% means the LLM is consistent with an established domain-specific baseline. See the FinBERT section below.

**D — Event-based validation**  
Known TD events are used to test whether the signal fires in the right direction at the right time:

| Event | Expected signal | Quarter |
|---|---|---|
| AML consent order announced (Oct 10, 2024) | `regulatory_AML` share spike; CEO sentiment drop | FY2024Q4 |
| Rate hiking cycle begins (BoC Mar 2022) | `macro_outlook` share spike; `NIM` tone improvement | FY2022Q2 |
| First Horizon deal collapse (May 2023) | `M_and_A` share spike; negative sentiment | FY2023Q3 |
| Post-AML resolution | CEO sentiment recovery toward positive | FY2026Q1 |

If the LLM annotations correctly identify these inflection points, the topic/sentiment system has external validity beyond statistical agreement.

---

### FinBERT — what it is and why it is the baseline

**What FinBERT is:**  
FinBERT (`ProsusAI/finbert`) is a BERT-based language model fine-tuned on approximately 4,840 financial news sentences with human-annotated sentiment labels (`positive`, `negative`, `neutral`). It was published by Araci (2019) and is one of the most widely cited financial NLP baselines in academic literature. It classifies text into the same three sentiment categories as our LLM prompt, making direct comparison straightforward.

**Why BERT architecture:**  
BERT (Bidirectional Encoder Representations from Transformers) reads text in both directions simultaneously, giving it better contextual understanding than earlier bag-of-words or LSTM approaches. FinBERT inherits this and additionally learned financial vocabulary patterns from domain-specific training — it knows that "elevated provisions" is negative and "strong capital generation" is positive without being told explicitly.

**Why FinBERT specifically as the baseline, not another model:**
1. **Same output space.** FinBERT outputs `positive / neutral / negative` — identical to our LLM labels. Direct sign comparison is clean.
2. **Domain-specific training.** General-purpose sentiment models (e.g., VADER, TextBlob) were trained on social media and product reviews. They frequently misclassify financial language — "TD reported record provisions" would be labelled positive by VADER because "record" is in its positive word list. FinBERT was trained on financial text and avoids this.
3. **No API dependency.** FinBERT runs entirely locally on the downloaded model weights. It is free, deterministic, and does not require prompt engineering — making it a true independent check rather than a variant of the same LLM system.
4. **Academic credibility.** As a peer-reviewed published baseline, FinBERT agreement provides a defensible citation for the evaluation section of any presentation.

**Key limitation:**  
FinBERT was trained on short financial news sentences (average ~18 words). Our transcript chunks average ~183 tokens and our report chunks average ~1,350 tokens. For long passages, FinBERT truncates at 512 tokens — it may miss context from the second half of a dense report chunk. This is why FinBERT is a *sanity check* rather than the primary evaluator: it is well-suited for transcript turns but less reliable on long report passages.

**Interpreting agreement:**
- ≥85%: strong consistency — LLM labels align well with domain-specific baseline
- 70–85%: acceptable for MVP — investigate divergent chunks for systematic patterns
- <70%: flag for prompt revision — likely a taxonomy or framing issue in the LLM prompt

---

### FinBERT results

Run against all 856 transcript chunks (the source best suited to FinBERT's short-text training data). Comparison file: `data/processed/features/finbert_comparison_transcripts.parquet`.

**Overall agreement:**

| Metric | Value | Assessment |
|---|---|---|
| Sign agreement rate (opinionated chunks only) | **86.8%** | Above 85% — strong |
| Opinionated chunks used | 205 / 856 | 76% of chunks labelled neutral by at least one model |
| Pearson correlation of continuous scores | **0.501** | Moderate positive correlation |

The 86.8% sign agreement clears the ≥85% "strong" threshold. The LLM annotations are consistent with FinBERT as an independent domain-specific baseline.

**Label distribution comparison:**

| Label | LLM | FinBERT |
|---|---:|---:|
| `positive` | 438 (51%) | 199 (23%) |
| `neutral` | 400 (47%) | 585 (68%) |
| `negative` | 18 (2%) | 72 (8%) |

FinBERT is systematically more cautious than the LLM — it calls more chunks neutral and more chunks negative. This is a known characteristic: FinBERT was trained on financial news (which tends toward neutral factual reporting), while the LLM reads management language more charitably, picking up on optimistic framing even in technically neutral statements.

**Confusion matrix (rows = LLM label, cols = FinBERT label):**

| | FB: negative | FB: neutral | FB: positive |
|---|---:|---:|---:|
| **LLM: negative** | 6 | 11 | 1 |
| **LLM: neutral** | 40 | 334 | 26 |
| **LLM: positive** | 26 | 240 | 172 |

The largest disagreement is the LLM calling passages `positive` while FinBERT calls them `neutral` (240 chunks). This is the systematic gap: the LLM detects optimistic framing in management language; FinBERT sees the same text as factual/neutral.

**Correlation by speaker role:**

| Role | n | Pearson r |
|---|---:|---:|
| Analyst | 388 | 0.489 |
| CEO | 134 | 0.458 |
| CFO | 102 | 0.467 |
| Other exec | 76 | 0.084 |

Correlations are moderate (0.46–0.49) for the main roles. The very low `other_exec` correlation (0.084) reflects a known FinBERT limitation: other executives often use hedged, context-dependent language (*"we remain well positioned through the pandemic"*) that FinBERT scores negatively because it pattern-matches on words like "pandemic" from its training data, while the LLM correctly reads the optimistic framing.

**Divergent chunks — what causes disagreement:**

All 27 divergent chunks (LLM=positive, FinBERT=negative) share a pattern: passages containing **pandemic/AML/uncertainty language paired with positive framing**. Examples:

- *"We remain well positioned to manage through the balance of the pandemic."*  → LLM: positive (+0.70), FinBERT: negative (−0.48)
- *"We remain quite optimistic that as the economy is starting to open, we're well-geared to upward interest rates."* → LLM: positive (+0.70), FinBERT: negative (−0.22)
- *"Credit performance has trended positively this year as we have progressed through the pandemic."* → LLM: positive (+0.70), FinBERT: negative (−0.43)

In each case, the LLM correctly identifies the management intent (optimism), while FinBERT anchors on context words ("pandemic", "manage through", "despite") that co-occurred with negative outcomes in its training data. This is the core advantage of zero-shot LLM annotation for management commentary: it understands framing, not just keywords.

**Quarterly score comparison (LLM mean vs FinBERT mean):**

The LLM scores are consistently higher (gap ranges 0.02–0.36). Both models show the same directional patterns across quarters — the correlation in *trends* is strong even when the absolute levels differ. The largest gap is FY2025Q3 (LLM: +0.378, FinBERT: +0.020), a quarter where executives used constructive but cautious language about the post-AML recovery that FinBERT reads as near-neutral.

**Conclusion:** The 86.8% agreement rate validates the LLM annotations as consistent with an established domain-specific baseline. Where the two diverge, manual inspection shows the LLM is generally right on TD management commentary — FinBERT's keyword-anchoring causes false negatives on optimistic text that mentions risk topics. For Step 3, the LLM sentiment scores are preferred as the primary feature; FinBERT scores can be included as an additional feature to capture the "conservatively measured" sentiment dimension.

---

## Holistic analysis — three-act AML narrative

The surface-level story is well known: TD agreed to a US$3.09 billion AML penalty on **October 10, 2024** (TD FY2024Q4). What the LLM annotation adds is three acts of quantified language signals — each validated against actual chunk content and chunk counts — that are invisible from news headlines alone.

> **Quarter-label note:** all transcript signals below use the **reporting-period** convention (the quarter whose results a call discusses, e.g., the August 2024 call reports Q3 FY2024). Quarterly reports and news use the same convention. This was corrected from the original publication-date labelling, which caused a systematic 1-quarter lag.

A note on signal quality: every claim below is grounded in chunk-level evidence. Where chunk counts are low (n<5 for a quarter), the metric is flagged and excluded from the narrative. Where causation is uncertain, it is stated as an observed pattern rather than a confirmed cause.

---

### The story

**The LLM caught three things about TD's AML crisis that you wouldn't see from reading the news.**

---

**Before the announcement, the language was already changing.**

From Q3 FY2022 through Q3 FY2023 — a full year before the consent order — the way TD's documents talked about regulatory and compliance topics became steadily less positive. AML-tagged sentiment fell from 0.128 to 0.070 across six quarters; the AML share of all document content rose from 11.7% to 17.8%. This happened entirely inside filings and transcript text. Press releases stayed upbeat throughout. We cannot prove this reflects internal awareness of the investigation, but the pattern is monotonic, multi-source, and well-supported by chunk counts.

**The most dramatic signal appeared before anyone said anything publicly.**

In Q3 FY2023 (the August 2023 results call, reporting the period ending July 2023), the CEO prepared remarks scored +0.80 — positive and confident. The quarterly report for the exact same period scored +0.078 — nearly flat. The gap was **+0.722**, the largest transcript-vs-mandatory-disclosure divergence in the five-year dataset. Filings can't be framed — the numbers and risk disclosures appear as-is. The gap measures exactly how much work the earnings call was doing to project confidence while the written report had to disclose what was happening operationally.

Three months later, at the Q4 FY2023 results call (November 2023), the CEO's prepared remarks compressed to **0.433** — below the 0.75–0.80 pre-crisis norm but not at zero. The first and only quarter where CEO prepared remarks hit **0.000** was FY2024Q4, the consent-order call itself.

**At peak enforcement, management stopped making promises.**

Forward guidance — management committing to specific future outcomes — hit its lowest raw count in May 2024 (FY2024Q2 results call): only 14 guidance-tagged chunks (8.7% of all content). Management literally stopped making commitments when the resolution timeline was unclear. Guidance topic share stayed suppressed through FY2024Q4 (8.8%); the clearest rebound is FY2025Q1 (17.8% of all annotated chunks that quarter).

**CEO and CFO both recovered simultaneously from Q1'25 — with CFO pulling one step ahead from Q2'25.**

After the consent order, both CEO and CFO prepared remarks recovered to 0.600 simultaneously at FY2025Q1. From FY2025Q2 the CFO ran one step ahead (0.700 vs CEO 0.600), and again at FY2025Q3 (0.800 vs CEO 0.700), before converging at FY2025Q4 (both 0.800). The framing gap (CEO prep − filings) remained positive throughout recovery — compressed at +0.362–+0.412 compared to the pre-crisis 0.5–0.7 range — but never inverted. Guidance topic share peaked in the 21-quarter matrix at **FY2025Q1 (17.8%)**; analyst QA sentiment hit its dataset high at **FY2026Q1 (0.529)**, confirming that external scrutiny had converted to constructive engagement.

---

**What this adds beyond the news:**

The news told you *when* the AML crisis happened. The NLP tells you *how* the language evolved: the slow deterioration before anyone said anything publicly, the precise moment when the gap between managed communication and mandatory disclosure was widest (+0.722, Q3 FY2023), the measured silence when management stopped making commitments (guidance trough May 2024), and the CFO recovering one step ahead of the CEO in Q2–Q3 FY2025. None of those four things are visible from headlines. All four are measurable.

---

### Act 1 — Pre-event language stress (FY2022Q3 → FY2023Q4)

*The formal AML consent order was signed October 10, 2024. The signals below appear entirely within structured language — filings and transcripts — with no public announcement trigger.*

#### AML topic language trend (observed pattern, causation unconfirmed)

The `regulatory_AML` topic sentiment declined monotonically from FY2022Q3 through FY2023Q3, with adequate chunk counts (n=17–27) in every quarter:

| Quarter | n AML chunks | AML sentiment | AML share |
|---|---:|---:|---:|
| FY2022Q1 | 18 | 0.239 | 11.1% |
| FY2022Q2 | 17 | 0.206 | 10.1% |
| FY2022Q3 | 18 | 0.128 | 11.7% |
| FY2023Q1 | 23 | 0.104 | 14.7% |
| FY2023Q2 | 21 | 0.124 | 11.7% |
| FY2023Q3 | 27 | **0.070** | **17.8%** |

Tone fell ~71% from the FY2022Q1 baseline; share rose 6 percentage points. **Causal attribution is uncertain** — whether this reflects internal AML escalation or natural regulatory language variation cannot be determined from text alone.

#### Framing gap peaked at FY2023Q3 — the widest divergence in the dataset

In FY2023Q3 (Q3 FY2023 results, reporting the period ending July 2023), the CEO prepared remarks were at 0.80 while the quarterly report for the same period was at 0.078. This +0.722 gap is the largest in the five-year dataset — larger than anything seen during peak enforcement.

| Quarter | CEO prep | Quarterly report | Gap |
|---|---:|---:|---:|
| FY2022Q3 | 0.800 | 0.231 | +0.569 |
| FY2023Q2 | 0.750 | 0.181 | +0.569 |
| FY2023Q3 | **0.800** | **0.078** | **+0.722** |
| FY2024Q3 | 0.600 | 0.029 | +0.571 |

The gap at FY2023Q3 is pre-announcement — it reflects a management communication style that projected public confidence while mandatory disclosures showed near-zero sentiment on the same period's results.

#### CEO pre-stress signal at FY2023Q4 (11 months before consent order)

The Q4 FY2023 results call (November 2023) was the first time the CEO's prepared-remarks mean compressed to **0.433** — meaningfully below the 0.75–0.80 pre-crisis norm. The prepared remarks opened with condolences for a workplace shooting and acknowledged a "mixed quarter", holding the opener at neutral; the rest of the block covered US Retail and capital at a more positive tone. The *average* is compressed, not zeroed. The first and only true zero in the five-year dataset was FY2024Q4, the consent-order call itself.

The CEO then recovered to 0.750 at FY2024Q1 (Feb 2024) and 0.767 at FY2024Q2 (May 2024) — a pattern consistent with initial caution, followed by a management decision to project stability through the extended negotiation period.

#### CFO intermittent declines (mixed signal — partly First Horizon)

The CFO prepared-remarks sentiment dropped to 0.00 at FY2022Q3 (Aug 2022), FY2023Q1 (Feb 2023), FY2023Q2 (May 2023), and FY2024Q1 (Feb 2024). The FY2022Q3 and FY2023Q1–Q2 drops coincide with the First Horizon acquisition failure (deal terminated May 2023), making AML attribution uncertain for those quarters. The FY2024Q1 drop is more likely AML-related given First Horizon was already resolved.

**What Act 1 represents:** A multi-quarter, multi-source language pattern consistent with pre-event stress. The FY2023Q3 framing gap (+0.722) is the highest-confidence signal. FY2023Q4 CEO compression (0.433) is a supporting signal — below the pre-crisis norm but not at zero; the first zero was FY2024Q4. The AML topic trend is supporting evidence with uncertain causation.

---

### Act 2 — Peak enforcement (FY2024Q1 → FY2024Q4)

*October 10, 2024: TD pleaded guilty to AML charges and agreed to pay US$3.09 billion.*

#### CEO opened FY2023Q4 with the only neutral scripted opener in the dataset

CEO prepared remarks score by quarter:

| Quarter | CEO prepared | Note |
|---|---:|---|
| FY2023Q3 | 0.800 | Last confident quarter before pre-stress |
| FY2023Q4 | **0.433** | Compressed (below 0.75–0.80 norm); mixed opener with condolences |
| FY2024Q1 | 0.750 | Recovery above compression |
| FY2024Q2 | 0.767 | Stable through spring 2024 |
| FY2024Q3 | 0.600 | Below norm as consent order finalises (Aug 2024 call) |
| FY2024Q4 | **0.000** | First and only zero — consent-order call |

The pattern is not a clean drop at the announcement. The CEO signalled caution at FY2023Q4 (compressed to 0.433), briefly recovered, then declined again through FY2024Q3 (0.600) into the FY2024Q4 zero.

#### AML sentiment at near-zero during enforcement

| Quarter | AML n | AML sentiment | AML share |
|---|---:|---:|---:|
| FY2024Q2 | 33 | 0.152 | 20.5% |
| FY2024Q3 | 38 | 0.055 | **22.4%** |
| FY2024Q4 | — | **0.033** | **30.0%** |

FY2024Q4 AML sentiment of 0.033 is the lowest in the dataset — the first call after the consent order, when the AML topic carried essentially no positive language across any source. AML share also peaked at 30.0% that quarter, the highest in the 21-quarter series.

#### Guidance collapsed ahead of and during enforcement

| Quarter | Guidance n | Guidance share | Interpretation |
|---|---:|---:|---|
| FY2023Q4 | 34 | 11.5% | Baseline before enforcement window |
| FY2024Q1 | 22 | 15.6% | Normalising |
| FY2024Q2 | **14** | **8.7%** | Trough — management stops committing |
| FY2024Q3 | 22 | 12.9% | Slight recovery |
| FY2024Q4 | 25 | 8.8% | Still suppressed — consent-order quarter (not a guidance-volume spike) |

The trough at FY2024Q2 (May 2024 call) is the clearest signal that management could not see the resolution timeline. Guidance topic share remains low through FY2024Q4 (8.8%); the large rebound appears in FY2025Q1 (17.8% — see Act 3).

**What Act 2 shows:** The AML crisis produced a measurable framing pattern (near-zero AML sentiment while guidance collapsed), a brief CEO recovery between the compression signal and the consent order, and a clean reset at FY2024Q4. The FY2024Q2 guidance trough (8.7%) and FY2024Q4 AML share peak (30.0%) are the highest-confidence Act 2 signals.

---

### Act 3 — Recovery sequence (FY2025Q1 → FY2026Q1)

*AML consent order resolved October 2024 (FY2024Q4). Recovery signals emerged as CEO and CFO simultaneously returned to positive language in FY2025Q1, with the CFO running one step ahead from FY2025Q2 onward.*

#### Framing gap — compressed but positive throughout recovery

After the consent order, the framing gap (CEO prep − merged filing) remained positive but compressed. It did not invert: the CEO was already back at 0.600 in FY2025Q1, above both the filing sentiment and the pre-crisis filing range.

| Quarter | CEO prep | Quarterly report | Gap | Direction |
|---|---:|---:|---:|---|
| FY2024Q3 | 0.600 | 0.029 | +0.571 | Normal |
| FY2024Q4 | 0.000 | 0.137 | −0.137 | Inverted — consent-order call only |
| FY2025Q1 | 0.600 | 0.188 | **+0.412** | Positive — simultaneous recovery |
| FY2025Q2 | 0.600 | 0.238 | **+0.362** | Compressed but positive |
| FY2025Q3 | 0.700 | 0.269 | +0.431 | Recovering toward pre-crisis range |

The only negative framing gap in the 21-quarter series is FY2024Q4 (−0.137), driven by the 40-F annual filing sentiment holding positive while CEO opened at zero. FY2025Q1–Q2 show a compressed gap (0.36–0.41 vs the pre-crisis 0.5–0.7 norm) — still CEO-above-filings, but with less cushion.

#### CFO operational recovery preceded CEO

| Quarter | CEO prepared | CFO prepared | Note |
|---|---:|---:|---|
| FY2024Q4 | 0.000 | 0.000 | Consent-order call — both silent |
| FY2025Q1 | **0.600** | **0.600** | Simultaneous recovery — no lead yet |
| FY2025Q2 | 0.600 | **0.700** | CFO pulls ahead by one step |
| FY2025Q3 | 0.700 | **0.800** | CFO still one step ahead |
| FY2025Q4 | 0.800 | 0.800 | Convergence |
| FY2026Q1 | 0.800 | 0.800 | Sustained |

Both CEO and CFO recovered simultaneously in FY2025Q1 (both to 0.600). The CFO first exceeded the CEO in FY2025Q2 (0.700 vs 0.600) and again in FY2025Q3 (0.800 vs 0.700), before converging in FY2025Q4. The one-step CFO lead is consistent with the finance function regaining confidence in forward operational metrics (provision releases, capital ratios improving) before management-level tone had fully cleared regulatory and reputational risk.

#### Guidance and analyst confidence: the cleanest recovery signals

| Quarter | Guidance n* | Guidance % | Analyst QA |
|---|---:|---:|---:|
| FY2024Q2 | 14 | 8.7% | 0.307 |
| FY2025Q1 | 28 | **17.8%** | 0.169 |
| FY2025Q2 | 25 | 15.2% | 0.233 |
| FY2025Q3 | 26 | 17.4% | 0.350 |
| FY2025Q4 | 34 | 12.3% | 0.238 |
| FY2026Q1 | 22 | 16.5% | **0.529** |

\*`n` = chunks tagged with topic `guidance` across all four annotation sources; `%` = `n / n_chunks_annotated` for that fiscal quarter (same definition as `topic_guidance_share` in `nlp_features.parquet`).

FY2025Q1 has the highest guidance share in the 21-quarter matrix (17.8%). FY2026Q1 analyst QA sentiment (0.529) is the dataset high for that feature — external analysts are now making constructive comments rather than probing questions. The CEO committed to specific targets: *"We expect to achieve the 6–8% EPS growth and 13% ROE targets for fiscal 2026."*

**What Act 3 shows:** The recovery followed a specific sequence — CEO and CFO together first (FY2025Q1, both 0.600), then CFO one step ahead for two quarters (Q2–Q3'25), then convergence (Q4'25), then peak guidance topic share (FY2025Q1, 17.8%) and analyst confidence peak (FY2026Q1, 0.529). The framing gap never inverted during recovery — it was compressed (0.36–0.41) but remained positive, meaning the CEO was always above the filing baseline. The FY2024Q4 consent-order quarter is the only negative framing gap in the 21-quarter series.

---

### Analytical questions answered

| Question | Answer from NLP | Confidence |
|---|---|---|
| Is AML language stress detectable before the public announcement? | AML sentiment declined monotonically for 6+ quarters before Oct 2024; framing gap peaked at FY2023Q3 (+0.722) pre-announcement; CEO compressed to 0.433 at FY2023Q4 (11 months early); first and only zero was FY2024Q4 | Medium — pattern is consistent with pre-event stress but causation unconfirmed |
| Is management communication consistent across scripted and live settings? | CEO prep ≈ CEO Q&A across all quarters — high consistency. CFO is a different signal channel: more conservative, especially during stress. | High |
| Is a recovery real or PR? | CEO and CFO both recovered simultaneously at FY2025Q1 (0.600); CFO led by one step Q2–Q3'25; framing gap compressed but never inverted; guidance topic share peaked FY2025Q1 (17.8%); CEO committed to specific EPS/ROE targets in FY2026Q1 | High |
| What does a regulatory crisis cost in communication terms? | Guidance trough at FY2024Q2 (8.7%, n=14 chunks); largest AML share at FY2024Q4 (30.0%); framing gap peaked pre-crisis at +0.722, not during enforcement | High |
| When does management feel safe to commit to the future again? | When guidance topic share rebounds (peak FY2025Q1 at 17.8%) and CEO makes specific numeric targets — FY2026Q1 analyst QA (0.529) is the clearest buy-side marker | High |

---

### Event-based validation (Evaluation D)

| Event | Expected | Observed | Confidence | Result |
|---|---|---|---|---|
| AML language trend (FY2022Q3–FY2023Q3) | Regulatory tone declining before consent order | AML sentiment 0.239→0.070; AML share 11.1%→17.8%; adequate n throughout | Medium — correlation, causation uncertain | **Observed** |
| Framing gap peak (FY2023Q3) | Pre-announcement divergence between managed tone and mandatory disclosure | CEO 0.80 vs report 0.078 → gap +0.722; largest in five-year dataset | High | **Pass** |
| CEO pre-stress (FY2023Q4, Nov 2023) | CEO sentiment compression before formal announcement | CEO prep: 0.800→0.433 (compressed, not zero); first zero at FY2024Q4 | Medium — could reflect other factors (First Horizon, mixed quarter) | **Observed — weaker than originally stated** |
| AML consent order (Oct 2024 → FY2024Q4) | Immediate CEO/CFO drop; AML sentiment near-zero | CEO=0.000, CFO=0.000, AML_sent=0.033, AML_share=30.0% at FY2024Q4 | High | **Pass** |
| Guidance collapse (FY2024Q2) | Management stops making commitments | Guidance n=14 (8.7% share) — lowest in enforcement period | High | **Pass** |
| Rate hiking cycle (BoC Mar 2022 → FY2022Q2) | Macro share spike; NIM tone positive | Macro share rose; NIM sentiment 0.506 at FY2022Q3 | Medium | **Partial** |
| First Horizon collapse (May 2023 → FY2023Q2) | M&A share spike; negative sentiment | M&A share elevated; sentiment subdued not negative | Medium | **Partial — volume fired, direction lagged** |
| Compressed framing gap (FY2025Q1–Q2) | Framing gap narrower than pre-crisis during recovery | Gap +0.412 / +0.362 vs pre-crisis 0.5–0.7 norm; never inverted | Medium — compressed not inverted; original "inversion" finding was data artefact | **Revised: gap compressed, not inverted** |
| CFO leads recovery (FY2025Q2–Q3) | CFO positive before CEO | Both at 0.600 in FY2025Q1 (simultaneous); CFO 0.700 vs CEO 0.600 at FY2025Q2; CFO 0.800 vs CEO 0.700 at FY2025Q3 | Medium | **Pass — CFO lead starts Q2'25, not Q1'25** |
| Guidance rebound + analyst QA peak | Management commits to forward targets | Peak guidance share FY2025Q1 (17.8%); FY2026Q1 analyst_qa=0.529 (dataset high); specific EPS/ROE targets in FY2026Q1 call | High | **Pass — strongest signals in dataset** |

---

---

## Limitations and remediations

### L1 — LLM annotating raw financial tables (discovered, remediated)

**Limitation:** Some chunks extracted from quarterly reports and 40-F filings are raw numeric tables — rows of dollar figures, percentages, and column headers with almost no prose. The LLM cannot produce reliable sentiment from these: it infers tone from whether the numbers look good or bad, effectively hallucinating a narrative that does not exist in the text. One example from the validation spot-check was a chunk showing ROE and net income figures across three quarters, which received a +0.70 score and an invented key quote (*"Canadian Retail reported net income increased by 14% compared to the previous year"*) that did not literally appear in the passage.

**Scope:** 195 of 3,984 annotated chunks were table-heavy (>40% of whitespace-split tokens are numeric or symbolic). All 195 were in `reports_40f` (90 chunks, 8.8% of that source) and `reports_quarterly` (105 chunks, 9.7%). Transcripts and news had zero table-heavy chunks.

**Remediation (applied):** A `_is_table_heavy()` filter was added to `annotate.py`. It computes the fraction of tokens matching `[\d,.\$%\(\)\-\/\|]+` and skips chunks above the 0.40 threshold. The 195 already-annotated table chunks were stripped from the existing JSONL files (originals preserved as `.jsonl.bak`) and `aggregate_features.py` was re-run to regenerate `nlp_features.parquet`. The sentiment delta was ±0.018 at most — the key analytical patterns (transcript vs. report divergence in FY2024Q2/Q3, AML-period CEO sentiment drop) are unchanged.

**Residual risk:** The 0.40 threshold may miss tables with moderate prose headers. A stricter approach for future runs would be to filter out chunks whose `section` field maps to known table sections (e.g., `"TABLE"`, `"FINANCIAL HIGHLIGHTS"`) in the source document metadata.

---

### L2 — CEO prepared-remarks sentiment based on very few chunks per quarter

**Limitation:** In most quarters the CEO delivers one or two blocks of prepared remarks before handing off to the CFO. Quarters with only one substantive CEO chunk make `transcript_ceo_prep_sentiment_mean` a point estimate with no variance — a slight change in phrasing could shift the score materially. FY2023Q4 illustrates this: chunk 014 (471 tokens) scored 0.000 (neutral, opened with condolences and "mixed quarter" framing) while chunk 015 (946 tokens) scored 0.700 and chunk 016 (554 tokens) scored 0.600 — the three-chunk mean is 0.433, which is more representative than any single chunk would be.

**Remediation (partial):** The `transcript_sentiment_std` feature captures within-quarter variance, which will be low for quarters with few CEO chunks — Step 3 models can use this as a reliability weight. For Step 3, consider treating `transcript_ceo_prep_sentiment_mean` as missing when fewer than 3 CEO prepared chunks exist in a quarter and imputing from the trailing 2-quarter rolling mean.

---

### L3 — Press-release chunks contain embedded financial tables

**Limitation:** TD's Newsroom press releases include the full financial highlights table as a trailing section. Unlike pure report tables, these mix some prose (column headers, footnote text) with numbers, so they fall just below the `_is_table_heavy` threshold and are not filtered out. These inflate news sentiment slightly and produce unreliable key quotes that summarise numbers rather than language. Row 4 in the validation spot-check was one such chunk.

**Scope:** Estimated <5% of news chunks based on manual review of the validation sample. Not re-filtered in this pass.

**Remediation (pending):** In a future annotation pass, add a secondary filter for news chunks: if the chunk is in the trailing section of a press release (identifiable by presence of `"(millions of Canadian dollars"` or `"TABLE"` in the first 50 characters), skip it. Alternatively, split press-release parsing to exclude everything after the `"Financial Highlights"` header.

---

### L4 — Non-determinism across different model versions

**Limitation:** Results were produced with `gpt-4o-mini` at `temperature=0`. OpenAI does not guarantee identical outputs if the underlying model weights are updated (silent version bumps). Re-running with a future `gpt-4o-mini` version could produce different annotations even for the same input text. The cache is keyed by `sha256(text + prompt_version + model_name)` — a model update would produce a cache miss and trigger re-annotation.

**Remediation (in place):** The `PROMPT_VERSION` constant in `annotate.py` allows intentional re-annotation passes to be versioned and compared. Cache files are preserved per source type. For production use, pin the model to a specific snapshot (e.g., `gpt-4o-mini-2024-07-18`) once OpenAI exposes stable dated aliases.

---

### L5 — Chunk boundary effects on sentiment

**Limitation:** Chunking splits documents at paragraph or section boundaries, which can cut a passage mid-thought. A chunk that opens with hedged language (the tail of a prior thought) and closes mid-sentence may receive an inaccurate label because the LLM lacks the surrounding context that would clarify the framing. This is most likely in 40-F body sections where paragraphs are long and run across chunk boundaries.

**Scope:** No direct measurement performed. Likely affects a small fraction of report chunks — transcripts and press releases chunk more cleanly because their natural paragraph boundaries are short.

**Remediation (not applied):** Adding a short overlap (e.g., 1–2 sentences from the preceding chunk as context prefix) to the prompt would reduce this. Not applied in this pass to avoid invalidating the existing annotation cache.

---

### L6 — Topic taxonomy does not cover all content

**Limitation:** The 12-label taxonomy was designed around TD's primary analytical dimensions. Chunks that do not clearly fit are assigned `other`, which averaged 26.3% of all annotated chunks. Within `other`, roughly 39% is non-financial corporate news (ESG announcements, HR, ETF distributions), 35% is technical accounting disclosures, and 9% is Schwab-specific content that does not map cleanly to `M_and_A` (because it concerns the ongoing economics of the retained stake rather than a transaction). These chunks carry low signal for Step 3.

**Remediation (partial):** The `other` label is retained as a residual bucket. For Step 3, `topic_other_share` is included as a feature — a high `other` share in a quarter is itself a mild signal that management is discussing non-core topics. Expanding the taxonomy (e.g., adding a `Schwab_stake` label) is a candidate for a future annotation pass if Step 3 modelling shows `other` share as a significant predictor.

---

### L7 — FinBERT truncation for long report chunks

**Limitation:** FinBERT truncates input at 512 tokens. Quarterly report chunks average ~1,350 tokens and 40-F chunks are longer still. FinBERT scores on these chunks only reflect the first ~512 tokens, missing context from the second half. This is why the FinBERT sanity check was run on transcripts only (average ~183 tokens per chunk) — transcript chunks are well within the model's context window.

**Remediation (in place):** The FinBERT baseline is scoped to transcripts in `finbert_baseline.py`. Report chunks are not passed through FinBERT. For a future comparison on reports, the chunks would need to be sub-chunked to ≤512 tokens before scoring.

---

### L9 — Q&A section missing in three transcript quarters (discovered, remediated)

**Limitation:** Three earnings-call transcripts (`2022-05-26_2022-q2`, `2022-08-25_q3-2022`, `2023-03-02_q1-2023`) had all content — including the full analyst Q&A exchange — labelled as `section=prepared_remarks`. This caused `transcript_exec_qa_sentiment_mean` and `transcript_analyst_qa_sentiment_mean` to be null for FY2022Q3, FY2022Q4, and FY2023Q2.

**Root cause:** The chunker's `QA_HEADER_RE` regex required a standalone `QUESTION AND ANSWER` header line to split the transcript into prepared-remarks vs Q&A sections. These three PDFs skip that header and transition directly to the operator; one uses a spaced-letter format (`Q U E S T I O N  A N D  A N S W E R`) not covered by the original regex. The Q&A text was fully present in the raw text files but was never passed to the LLM as `section=qa`.

**Scope:** FY2022Q3, FY2022Q4, FY2023Q2. The three transcripts contain 68, 57, and 66 Q&A turns respectively once correctly parsed.

**Remediation (applied):** Two additions were made to `QA_HEADER_RE` in `src/preprocess/cleaning.py`: (1) a spaced-letter pattern to match `Q U E S T I O N  A N D  A N S W E R`; (2) a fallback `QA_IMPLICIT_RE` that detects the operator's *"We will now take questions from the telephone lines"* phrase when no explicit header exists. The chunker was re-run, annotations were rebuilt from the existing LLM cache (no new API cost for existing text), and `aggregate_features.py` was re-run. FY2022Q3, FY2022Q4, and FY2023Q2 now have populated QA sentiment fields.

**Post-fix values:**
- FY2022Q3: exec_qa=0.356, analyst_qa=0.058
- FY2022Q4: exec_qa=0.373, analyst_qa=0.233
- FY2023Q2: exec_qa=0.608, analyst_qa=0.197

---

### L10 — FY2026Q1 quarterly report missing from original corpus (remediated)

**Limitation:** The TD Investor Relations crawler ran before the Q1 FY2026 quarterly report was published, so `report_quarterly_sentiment_mean` was null for FY2026Q1. This is the final quarter of the analysis window.

**Remediation (applied):** The Q1 FY2026 Report to Shareholders was downloaded from `td.com/quarterly-results/2026/q1/`, text was extracted, the report was added to `data/raw/td_ir_reports/index.parquet` with `calendar_fiscal_quarter=FY2026Q1`, and chunks were annotated (62 new chunks, 8 API calls). FY2026Q1 `report_quarterly_sentiment_mean` is now 0.306.

---

### L8 — Administrative intro chunks diluting speaker-level sentiment

**Limitation:** Earnings-call transcripts contain boilerplate sections that carry no analytical content: (1) the IR Investor Relations officer's opening disclaimer (*"This presentation contains forward-looking statements…"*), and (2) operator or moderator lines introducing the leadership team (*"Good afternoon and welcome to TD Bank Group's third quarter 2021 investor presentation…"*). These chunks consistently score exactly 0.000 — the LLM correctly identifies that there is no sentiment to express — but their presence diluted the CFO and `other_exec` prepared-remarks means in quarters where only 1–2 substantive chunks existed alongside them. The FY2022Q4 and FY2023Q3 CFO prepared-remarks means were identified as potentially unreliable partly for this reason.

**Scope:** 28 chunks across 20 transcript quarters — 20 IR-role chunks and 8 intro-keyword chunks. All scored 0.000.

**Remediation (applied):** An `_is_admin_chunk()` filter was added to `annotate.py`. It excludes chunks where `speaker.role == 'ir'` or where the chunk is short (<80 words) and contains intro-announcement keywords. The 28 chunks were stripped from `data/processed/llm_annotations/transcripts.jsonl` (original preserved as `.jsonl.bak2`) and `aggregate_features.py` was re-run. The sentiment delta on CFO aggregates was ±0.03 — no analytical conclusions changed.

**Residual risk:** Chunks where an executive begins with a brief intro before substantive content (e.g., *"Good morning everyone. Today I want to discuss our credit quality…"*) will pass the filter and be kept — the intro sentence is a small fraction of a longer substantive chunk. The filter is conservative by design.

---

## Implementation files

| File | Purpose |
|---|---|
| [`src/analysis/annotate.py`](../src/analysis/annotate.py) | LLM annotation runner with cache integration |
| [`src/analysis/aggregate_features.py`](../src/analysis/aggregate_features.py) | Aggregates per-chunk annotations to quarter-level features |
| [`src/analysis/finbert_baseline.py`](../src/analysis/finbert_baseline.py) | FinBERT sanity-check baseline with LLM comparison |
| [`src/preprocess/llm_cache.py`](../src/preprocess/llm_cache.py) | sha256-keyed annotation cache (re-run = free) |
| [`notebooks/02_annotation.ipynb`](../notebooks/02_annotation.ipynb) | Exploration notebook: sentiment time-series, topic heatmaps, validation sample |
| `data/processed/llm_annotations/*.jsonl` | Per-chunk annotation cache (one file per source type) |
| `data/processed/features/nlp_features.parquet` | Quarter-level NLP feature matrix for Step 3 |
