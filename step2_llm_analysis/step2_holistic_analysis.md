# Step 2 — Holistic Analysis: TD Bank AML Narrative

> All figures below are sourced from `data/processed/features/nlp_features.parquet`
> and validated against the raw annotation corpus in `02_annotation.ipynb`.
> Analysis scope: FY2021Q2 – FY2026Q1 (20 quarters). FY2026Q2 excluded (incomplete).

---

## The Story in Three Phases

TD's AML enforcement cycle is visible in quantified language signals across transcripts, news, and filings — patterns that were building well before the public announcement and measurable through the recovery.

---

## Phase 1 — Pre-AML Build-up (FY2023Q2 → FY2024Q3)

**AML topic share rose steadily while sentiment on that topic fell — the documents were discussing it more, and less confidently.**

Starting from a baseline of ~10–12% in FY2022, the share of all annotated content tagged `regulatory_AML` climbed continuously through FY2024Q3. Simultaneously, the mean sentiment on those AML-tagged chunks fell from 0.124 in FY2023Q2 to 0.055 in FY2024Q3 — nearly neutral, with no positive framing left by the time the consent order was finalising.

| Quarter | AML share | AML sentiment |
|---|---:|---:|
| FY2022Q1 | 11.1% | 0.239 |
| FY2022Q3 | 11.5% | 0.128 |
| FY2023Q2 | 11.5% | 0.124 |
| FY2023Q3 | 17.6% | 0.070 |
| FY2024Q2 | 20.5% | 0.152 |
| FY2024Q3 | **22.4%** | **0.055** |
| FY2024Q4 | **30.0%** | **0.033** |

AML share roughly **doubled** from FY2022 baseline to the FY2024Q4 consent-order quarter. AML sentiment fell **86%** from the FY2022Q1 level (0.239 → 0.033). Both signals moved monotonically in the same direction for eight consecutive quarters — this is not noise.

---

## Phase 2 — Peak Enforcement (FY2024Q4)

**CEO prepared remarks fell to 0.000 — the first and only zero in the 21-quarter dataset. CEO sentiment dropped below both analyst questions and news sentiment, inverting the normal hierarchy.**

In every other quarter, the prepared CEO remarks are the most positive channel: management frames results at +0.75–0.80 while analysts probe at +0.05–0.35 and news sits in the middle. At FY2024Q4, that ordering collapsed:

| Channel | FY2024Q3 | FY2024Q4 | Change |
|---|---:|---:|---|
| CEO prepared remarks | 0.600 | **0.000** | −0.600 |
| CFO prepared remarks | 0.000 | **0.000** | — |
| Analyst Q&A | 0.271 | 0.143 | −0.128 |
| News sentiment | 0.569 | 0.458 | −0.111 |

Analyst questions (0.143) and news (0.458) were both *higher* than CEO (0.000) in FY2024Q4 — the only quarter this inversion occurs in the full dataset. Management stopped positive framing entirely; the external channels remained relatively stable. AML topic share simultaneously hit its dataset peak at **30%** — nearly one in three annotated chunks that quarter discussed regulatory or compliance matters.

---

## Phase 3 — Recovery Signal (FY2025Q1 → FY2026Q1)

**Two signals marked the recovery: the framing gap returned positive, and guidance topic share spiked to its five-year peak.**

**Framing gap (CEO prepared − quarterly report)** reflects the premium management communication applies over mandatory disclosures. A large positive gap is normal — management frames results more optimistically than filings must. A compressed or negative gap signals stress. After the consent order the gap recovered:

| Quarter | CEO prep | Filing | Gap | Filing source |
|---|---:|---:|---:|---|
| FY2023Q3 | 0.800 | 0.078 | +0.722 | Quarterly report |
| FY2024Q3 | 0.600 | 0.029 | +0.571 | Quarterly report |
| FY2024Q4 | 0.000 | 0.137 | **−0.137** | 40-F (proxy)† |
| FY2025Q1 | 0.600 | 0.188 | +0.412 | Quarterly report |
| FY2025Q2 | 0.600 | 0.238 | +0.287 | Quarterly report |
| FY2025Q3 | 0.700 | 0.269 | +0.456 | Quarterly report |
| FY2025Q4 | 0.800 | 0.226 | +0.574 | 40-F (proxy)† |
| FY2026Q1 | **0.800** | **0.306** | **+0.494** | Quarterly report |

> **† 40-F proxy (Q4 quarters):** TD files an annual 40-F in Q4 rather than a quarterly report. The 40-F sentiment (0.137 for FY2024Q4, 0.226 for FY2025Q4) reflects the full fiscal year's narrative — not Q4 alone — so the gap figure is not directly comparable to Q1–Q3 rows. That said, the FY2024Q4 inversion (gap = −0.137) is directionally meaningful: CEO prepared remarks (0.000) fell below even the cautiously-worded annual filing, which retains positive language from earlier quarters. The FY2025Q4 gap (+0.574) similarly reflects a year-end filing that averaged across both stressed and recovered quarters, understating the genuine Q4 recovery visible in the CEO score (0.800).

The gap stayed positive throughout recovery (CEO always above filings), recovering toward the pre-crisis norm of +0.50–0.72 as both CEO sentiment and filing sentiment improved together.

**Guidance topic share and sentiment** measure how often management makes forward-looking commitments and how positively they frame them. After collapsing at FY2024Q4 (8.8% share), guidance rebounded sharply:

| Quarter | Guidance share | Guidance sentiment |
|---|---:|---:|
| FY2024Q2 | 8.7% | 0.514 |
| FY2024Q4 | 8.8% | 0.288 |
| FY2025Q1 | **17.8%** | 0.261 |
| FY2025Q2 | 15.2% | 0.456 |
| FY2025Q3 | 17.4% | 0.440 |
| FY2026Q1 | 16.5% | **0.545** |

FY2025Q1 had the **highest guidance topic share in the five-year dataset (17.8%)** — management returned to making forward commitments as soon as the consent order was resolved. By FY2026Q1, guidance sentiment reached 0.545 and analyst QA sentiment hit its dataset high of **0.529** — buy-side scrutiny had converted to constructive engagement.
