# Data Sources & Preprocessing — TD Analysis MVP

This document satisfies the guideline requirement to "clearly document your data sources and any preprocessing steps." It is updated as collectors run; the companion `[manifest.csv](../data/manifest.csv)` carries per-artifact provenance (URL, retrieval date, SHA-256, record count).

> **Note:** This document covers all data collected. Not all of it feeds the predictive model. Macro series, peer bank prices, and filing PDFs are collected for completeness and context; the final model inputs are a subset defined in `step3_predictive_model/final_model/docs/feature_engineering.md`.

## Scope

- **Subject**: Toronto-Dominion Bank (TSX: `TD.TO`, CAD)
- **Window**: 2021-01-01 → present (uniform 5-year lookback across all sources)
- **Canadian-issuer note**: TD does **not** file SEC 10-K / 10-Q. As a Foreign Private Issuer it publishes:
  - **Annual Report on Form 40-F** (10-K equivalent) — one per fiscal year
  - **Quarterly Report to Shareholders** (10-Q equivalent) — Q1-Q3 only; Q4 is rolled into the 40-F
  For this MVP we pull both directly from TD Investor Relations rather than from SEC EDGAR: TD IR is the canonical publisher, the narrative layout is identical, and it avoids the iXBRL/exhibit noise the EDGAR filings carry. TD's fiscal year ends **October 31**.

## Current collection summary


| Source                                 | Artifacts                              | Notes                                                                       |
| -------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------- |
| yfinance (prices)                      | 9 tickers × ~1327 trading days         | TD.TO + 5 peers + 2 benchmarks + CADUSD                                     |
| BoC Valet (macro)                      | 7 series                               | policy rate, overnight avg, 2y/5y/10y GoC, USD/CAD, CPI all-items (monthly) |
| TD IR reports (Form 40-F)              | 5 annual 40-F PDFs                     | FY2021–FY2025, fiscal year-ends Oct 31                                      |
| TD IR reports (Report to Shareholders) | 16 quarterly PDFs                      | FY2021–FY2025 × Q1/Q2/Q3 + FY2026Q1 (Q4 is the 40-F)                        |
| TD IR transcripts                      | 21 quarterly earnings-call transcripts | 4/year × FY21–FY25 + FY26Q1 — full 5-year coverage                          |
| TD Newsroom (stories.td.com)           | 688 press releases                     | 2021-01-05 → 2026-04-16 (~120–140 / year)                                   |


See the live provenance log in `[manifest.csv](../data/manifest.csv)` for every file.

## Sources (detail)

### 1. Prices & volume — `yfinance`

- **URL**: [https://finance.yahoo.com](https://finance.yahoo.com)
- **Tickers**: `TD.TO` + `RY.TO`, `BNS.TO`, `BMO.TO`, `CM.TO`, `NA.TO` (Big Six peers), `^GSPTSE` (TSX Composite), `XFN.TO` (iShares Cdn Financials ETF), `CADUSD=X` (FX)
- **Cadence**: daily OHLCV + adjusted close
- **Output**: `step1_data_collection/data/raw/prices/{ticker}.parquet`

### 2. Macro — Bank of Canada Valet

- **URL**: [https://www.bankofcanada.ca/valet/docs](https://www.bankofcanada.ca/valet/docs)
- **Series**:
  - `V39079` — policy target rate (overnight)
  - `AVG.INTWO` — overnight average (market)
  - `BD.CDN.2YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD` — GoC benchmark yields
  - `FXUSDCAD` — USD→CAD
  - `V41690973` — CPI all-items (StatCan, mirrored by Valet)
- **Output**: `step1_data_collection/data/raw/macro/{label}.parquet`
- **Note**: FRED was intentionally skipped for MVP — BoC Valet covers Canadian monetary + inflation + yield-curve needs without an API key.

### 3. Annual Reports on Form 40-F — TD Investor Relations

- **URL**: [https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/annual-report-on-form-40-f](https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/annual-report-on-form-40-f)
- **Collected**: FY2021, FY2022, FY2023, FY2024, FY2025 (5 PDFs, ~1.2M chars each)
- **Equivalent to**: US 10-K (annual report + MD&A + financial statements + risk factors in one filing)
- **Period-end date**: TD fiscal year-end (October 31)
- **Output**: `step1_data_collection/data/raw/td_ir_reports/annual/fy{YYYY}_form_40f.{pdf,txt}`

### 4. Quarterly Report to Shareholders — TD Investor Relations

- **URL**: `https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/quarterly-results/quarterly-results-{year}`
- **Collected**: FY2021–FY2025 × Q1/Q2/Q3 + FY2026Q1 = 16 PDFs (~350–425K chars each). Q4 is not published separately — it's absorbed into the 40-F.
- **Equivalent to**: US 10-Q (unaudited interim results + MD&A)
- **Output**: `step1_data_collection/data/raw/td_ir_reports/quarterly/{YYYY}_q{n}_report_to_shareholders.{pdf,txt}`

### 5. Earnings call transcripts — TD Investor Relations

- **URL**: `https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/quarterly-results/quarterly-results-{year}`
- **Collected**: 21 quarterly earnings-call transcripts (FY2021Q1 through FY2026Q1, full 5-year coverage).
- **Scope note**: industry-conference transcripts (Barclays / Scotia / RBC / NBF / TD Cowen) and special-event calls (deal announcements, AML investigation update) were **intentionally excluded** from the MVP corpus. They are easy to re-enable later but were dropped here to keep the narrative corpus strictly "TD-on-TD" earnings-quarter commentary.
- **Date inference**: call date parsed from the PDF cover page, with a fallback regex for letter-spaced titles (e.g., `"M A R C H  0 2 ,  2 0 2 3"`).
- **Output**: `step1_data_collection/data/raw/transcripts/{yyyy-mm-dd}_{slug}.{pdf,txt}` + `index.parquet`

### 6. Press releases — TD Newsroom

- **URL**: [https://stories.td.com/ca/en/news](https://stories.td.com/ca/en/news) (redirected from newsroom.td.com)
- **Coverage**: 688 articles, 2021-01-05 → 2026-04-16
- **Output**: `step1_data_collection/data/raw/news/{yyyy-mm-dd}_{slug}.{html,json}` + `index.parquet`

## Preprocessing

Step 2 uses a zero-shot LLM for topic and sentiment analysis. All preprocessing choices flow from that goal: preserve the linguistic signal LLMs rely on, strip only mechanical noise that inflates token cost without adding meaning.

### Pipeline overview

```
Raw PDF / HTML / JSON
        │
        ▼
  1. Text extraction      pdfplumber (PDFs) · BeautifulSoup (HTML) · json.loads (news)
        │
        ▼
  2. Structural parsing   Speaker tagging + prepared_remarks/Q&A split  ← transcripts only
        │
        ▼
  3. Text cleaning        Boilerplate strip · whitespace normalisation  ← all sources
        │
        ▼
  4. Text chunking        Split at paragraph/sentence boundaries into ~1,800-token windows
                          and attach metadata (date, quarter, speaker).
                          Content annotation happens later in Step 2 via the LLM.
        │
        ▼
  JSONL chunks with metadata  → consumed by Step 2 (LLM annotation)
```

Implementation files: `[src/preprocess/cleaning.py](../src/preprocess/cleaning.py)` · `[src/preprocess/chunking.py](../src/preprocess/chunking.py)` · `[src/preprocess/build_chunks.py](../src/preprocess/build_chunks.py)`

---

### Step 1 — Text extraction


| Source                      | Method                                    | Notes                                |
| --------------------------- | ----------------------------------------- | ------------------------------------ |
| 40-F PDFs                   | `pdfplumber` page-by-page                 | ~1.2–1.3M chars per filing           |
| Report to Shareholders PDFs | `pdfplumber` page-by-page                 | ~350–425K chars per quarterly report |
| Transcript PDFs             | `pdfplumber` page-by-page                 | ~60–90K chars per call               |
| News articles               | `BeautifulSoup` (`<p>`, `<li>`, `<h2/3>`) | Headline prepended to body           |


Raw text is written to a `.txt` sidecar alongside every `.pdf` so re-runs skip re-extraction.

---

### Step 2 — Structural parsing (transcripts only)

Transcripts are parsed **before** any cleaning, because the cleaning step collapses line breaks that the parser uses to detect speaker boundaries.

**Speaker tagging** — each utterance gets `{speaker_name, role, affiliation}`:


| Role label   | Who                                                        |
| ------------ | ---------------------------------------------------------- |
| `ceo`        | Chief Executive Officer                                    |
| `cfo`        | Chief Financial Officer                                    |
| `other_exec` | Other named TD executive                                   |
| `ir`         | Investor Relations host / operator intro                   |
| `analyst`    | Sell-side analyst (buy-side questions classified the same) |
| `operator`   | Conference call operator (housekeeping lines)              |
| `other`      | Anything not matched by the above                          |


**Q&A split** — the transcript is divided into two named sections:


| Section            | Content                                             |
| ------------------ | --------------------------------------------------- |
| `prepared_remarks` | CEO/CFO opening commentary before analyst questions |
| `qa`               | Full analyst Q&A exchange                           |


The split triggers on the first occurrence of phrases like "Questions and Answers", "Q&A Session", or an operator line containing "your first question".

**Date inference** — the call date is read from the PDF cover page. TD's older PDFs use letter-spaced titles (`M A R C H  0 2 ,  2 0 2 3`), so a two-pass regex is used: normal pattern first, then whitespace-stripped fallback.

---

### Step 3 — Text cleaning (`clean_text`)

Applied to all sources, **per speaker turn** for transcripts (after structural parsing):


| What is stripped                                                      | Why                                                                                                 |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Repeated page headers / footers (e.g., "TD Bank Group | Q3 2024 | 3") | PDF artefacts, zero semantic value                                                                  |
| Safe-harbour / forward-looking-statement boilerplate                  | Identical paragraph repeated in every filing and transcript; adds ~500 tokens of noise per document |
| Legal disclaimer blocks                                               | Same                                                                                                |
| Runs of 3+ blank lines                                                | Formatting noise                                                                                    |
| Leading/trailing whitespace per paragraph                             | Normalisation                                                                                       |



| What is **kept**        | Why                                                        |
| ----------------------- | ---------------------------------------------------------- |
| Original casing         | LLMs use casing as a signal (proper nouns, emphasis)       |
| Punctuation             | Required for sentiment and sentence-boundary understanding |
| Numbers and percentages | Core financial facts                                       |
| Paragraph structure     | Preserved for downstream chunking                          |


---

### Step 4 — Text chunking

**Code:** `chunking.py → chunk_text()` and `make_chunk_records()`.

Splits text into windows small enough to send to an LLM, respecting paragraph then sentence boundaries. Each window gets metadata attached — content annotation happens in Step 2.

`chunk_text()` packs paragraphs greedily up to the token target, falls back to sentence splitting for oversized paragraphs, and merges any tiny trailing chunk into the preceding one.


| Parameter               | Value                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| Tokeniser               | `tiktoken` — `cl100k_base` (GPT-4 / Claude compatible)             |
| Target chunk size       | 1,800 tokens                                                       |
| Hard maximum            | 4,000 tokens                                                       |
| Split priority          | paragraph break (`\n\n`) → sentence break (`.!?` + capital letter) |
| Minimum merge threshold | Chunks under 200 tokens merged into preceding chunk                |


Each chunk record carries:

```
chunk_id      globally unique (source_type:doc_id:section:index)
source_type   news | reports_40f | reports_quarterly | transcripts
doc_id        file stem (reports/transcripts) or article slug (news)
date          ISO-8601 calendar date of the document
fiscal_quarter  TD fiscal quarter, e.g. FY2024Q2  (Oct 31 FY-end convention)
section       "body" for reports; "prepared_remarks"/"qa" for transcripts
speaker       {name, role, affiliation}  — null for non-transcript sources
text          cleaned chunk text
token_count   tiktoken count of text
url           source URL
```

**Real examples from `data/chunks/transcripts.jsonl`:**

```
chunk_id : transcripts:2021-02-25_...:prepared_remarks:013
quarter  : FY2021Q1
section  : prepared_remarks          ← CEO opening, before Q&A
speaker  : {name: "Bharat Masrani", role: "ceo"}
tokens   : 282
text     : "Thank you, Gillian. And thank you, everyone, for joining us today.
            It's been almost a year since the COVID-19 pandemic transformed
            our lives. As we continue to witness its uneven impacts..."

chunk_id : transcripts:2021-02-25_...:qa:001
quarter  : FY2021Q1
section  : qa                        ← analyst question in Q&A session
speaker  : {name: "Paul Holden", role: "analyst", affiliation: "CIBC World Markets"}
tokens   : 536
text     : "Thanks. Good morning. I have a couple of questions for you related
            to expenses. You highlighted the branch optimization in the quarter
            and the costs you're going to run through this quarter and next..."

chunk_id : transcripts:2021-02-25_...:qa:005
quarter  : FY2021Q1
section  : qa                        ← CFO response in Q&A session
speaker  : {name: "Riaz Ahmed", role: "cfo"}
tokens   : 724
text     : "Thank you, Meny. Look, I think there are three things that you
            should look at when looking at PTPP and operating leverage. First
            is the currency impact..."
```

Output: one JSONL per source type in `data/chunks/`:


| File                      | Chunks    | Tokens     | Avg tokens/chunk |
| ------------------------- | --------- | ---------- | ---------------- |
| `reports_40f.jsonl`       | 1,023     | 1,381,463  | 1,350            |
| `reports_quarterly.jsonl` | 1,151     | 1,567,378  | 1,361            |
| `news.jsonl`              | 1,019     | 1,176,263  | 1,154            |
| `transcripts.jsonl`       | 1,390     | 258,171    | 185              |
| **Total**                 | **4,583** | **~4.38M** |                  |


> **Note on transcript chunk size**: the low average (185 tokens) reflects per-turn chunking — each speaker utterance is its own chunk. Short participant-header stubs (< 50 tokens) are filtered at Step 2 before sending to the LLM.

## Token budget for Step 2

Rough input-only cost for a full LLM pass across the ~4.3M-token corpus:


| Model tier          | Rate     | Full corpus (~4.38M) |
| ------------------- | -------- | -------------------- |
| Haiku / gpt-4o-mini | ~$0.25/M | ~$1.10               |
| Sonnet / gpt-4o     | ~$3/M    | ~$13                 |


That is an order of magnitude smaller than the previous EDGAR-heavy corpus (~22M tokens), because:

- The SEC 6-K supplementary financial packs (huge table dumps) are gone — we now use TD's canonical Report to Shareholders instead.
- Industry-conference / special-event transcripts are excluded.

## LLM annotation cache

- Cache key = `sha256(chunk_text + prompt_version + model_name)`
- Output: JSONL per source type in `step2_llm_analysis/data/llm_annotations/`
- Re-runs with unchanged prompt/model are free; prompt iteration only re-annotates impacted chunks.

Implementation: `[src/preprocess/llm_cache.py](../src/preprocess/llm_cache.py)`.

## Alignment

Every artifact and every chunk carries both `calendar_date` (or `date`) and `fiscal_quarter` (via `src.utils.config.fiscal_quarter`, based on TD's October 31 fiscal year-end), plus — for reports and transcripts — an explicit `td_fiscal_quarter_hint` (`FY2024Q2` etc.) so Step 2 and Step 3 joins are trivial.

## Provenance

`[manifest.csv](manifest.csv)` has one row per artifact, with URL, retrieval timestamp UTC, SHA-256 checksum, and record count (where applicable). Columns: `source, artifact_type, identifier, path, url, retrieved_at, record_count, sha256, notes`.