# Step 1: Data Collection

Data collection and preprocessing for the TD Bank AI analysis project. All raw and processed data lives in the project-root `data/` folder; this folder contains the scripts that produced it.

---

## Folder structure

```
step1_data_collection/
├── README.md                        ← this file
└── src/
    ├── collectors/
    │   ├── prices.py                ← daily OHLCV for TD + peers + benchmarks (yfinance)
    │   ├── macro.py                 ← BoC Valet: yields, policy rate, CPI, FX
    │   ├── transcripts.py           ← earnings-call transcript PDFs from TD IR
    │   ├── td_ir_reports.py         ← Form 40-F and quarterly Report to Shareholders PDFs
    │   └── newsroom.py              ← press releases from TD Newsroom (stories.td.com)
    ├── preprocess/
    │   ├── cleaning.py              ← boilerplate strip, whitespace normalisation
    │   ├── chunking.py              ← semantic chunking (1 800-token target)
    │   ├── build_chunks.py          ← orchestrates cleaning → chunking → JSONL output
    │   └── llm_cache.py             ← SHA-256 keyed LLM annotation cache
    └── features/
        ├── build_daily_features.py  ← merges all sources → model_features_daily.parquet
        └── data_quality_check.py    ← coverage and integrity checks on the feature table
```

Shared utilities (`src/utils/config.py`, `src/utils/manifest.py`) and model code (`src/models/`) remain at the project root so Step 2 and Step 3 can import them without circular dependencies.

---

## Data location

```
data/
├── raw/
│   ├── prices/                      # daily OHLCV parquets (one per ticker)
│   ├── macro/                       # BoC Valet series (one parquet per series)
│   ├── transcripts/                 # PDF + TXT + index.parquet
│   ├── td_ir_reports/
│   │   ├── annual/                  # Form 40-F (FY2021–FY2025)
│   │   └── quarterly/               # Report to Shareholders (Q1-Q3 per FY)
│   └── news/                        # HTML + JSON + index.parquet (TD Newsroom)
└── processed/
    ├── chunks/                      # JSONL per source type (LLM-ready)
    └── features/
        └── model_features_daily.parquet   # final merged daily feature table
```

---

## Data sources

| Source | Landing page / API | What we collect |
|---|---|---|
| **Prices & volume** | [Yahoo Finance](https://finance.yahoo.com) via [`yfinance`](https://pypi.org/project/yfinance/) | Daily OHLCV + adj. close for `TD.TO`, `RY.TO`, `BNS.TO`, `BMO.TO`, `CM.TO`, `NA.TO`, `^GSPTSE`, `XFN.TO`, `CADUSD=X` — 9 tickers × ~1 327 trading days |
| **Macro — rates, FX, CPI** | [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) | Policy rate (`V39079`), overnight avg (`AVG.INTWO`), GoC 2Y/5Y/10Y yields, USD/CAD (`FXUSDCAD`), CPI all-items (`V41690973`) |
| **Annual reports (Form 40-F)** | [TD Investor Relations — Annual Report on Form 40-F](https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/annual-report-on-form-40-f) | FY2021–FY2025 (5 PDFs). Canadian-issuer equivalent of the US 10-K; published Oct 31 each year |
| **Quarterly reports** | [TD Investor Relations — Quarterly Results](https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/quarterly-results) | FY2021–FY2025 × Q1/Q2/Q3 = 15 PDFs. Q4 is not published separately — it is absorbed into the 40-F |
| **Earnings call transcripts** | [TD Investor Relations — Quarterly Results](https://www.td.com/ca/en/about-td/for-investors/investor-relations/financial-information/financial-reports/quarterly-results) | 20 quarterly transcripts (FY2021Q2 → FY2026Q1). Industry-conference and special-event calls excluded to keep the corpus strictly "TD-on-TD" |
| **Press releases** | [TD Newsroom](https://stories.td.com/ca/en/news) | 688 articles, 2021-01-05 → 2026-04-16 (~120–140 per year) |

> **Note on Canadian filings**: TD does **not** file SEC 10-K / 10-Q. As a Foreign Private Issuer it publishes the Form 40-F (annual) and Report to Shareholders (quarterly). Both are downloaded directly from TD Investor Relations rather than SEC EDGAR — TD IR is the canonical publisher and avoids iXBRL / exhibit noise.

---

## Preprocessing pipeline

All text sources go through a common five-step pipeline before being passed to the LLM in Step 2.

```
Raw PDF / HTML / JSON
        │
        ▼
  1. Text extraction      pdfplumber (PDFs)  ·  BeautifulSoup (HTML)  ·  json.loads (news JSON)
        │
        ▼
  2. Structural parsing   Speaker tagging + Q&A split          ← transcripts only, before cleaning
        │
        ▼
  3. Text cleaning        Boilerplate strip  ·  whitespace normalisation
        │
        ▼
  4. Section tagging      Regex split on known MD&A headings   ← reports only
        │
        ▼
  5. Semantic chunking    1 800-token target, section → paragraph → sentence boundaries
        │
        ▼
  JSONL chunks  →  data/processed/chunks/
```

### Step 1 — Text extraction

| Source | Method | Notes |
|---|---|---|
| Form 40-F PDFs | `pdfplumber` page-by-page | ~1.2–1.3 M chars per filing |
| Report to Shareholders PDFs | `pdfplumber` page-by-page | ~350–425 K chars per quarterly report |
| Transcript PDFs | `pdfplumber` page-by-page | ~60–90 K chars per call |
| News articles | `BeautifulSoup` (`<p>`, `<li>`, `<h2/3>`) | Headline prepended to body |

Raw text is written to a `.txt` sidecar alongside every `.pdf` so re-runs skip re-extraction.

### Step 2 — Structural parsing (transcripts only)

Transcripts are parsed **before** cleaning because the cleaning step collapses line breaks that the parser uses to detect speaker boundaries.

**Speaker tagging** — each utterance gets `{speaker_name, role, affiliation}`:

| Role | Who |
|---|---|
| `ceo` | Chief Executive Officer |
| `cfo` | Chief Financial Officer |
| `other_exec` | Other named TD executive |
| `analyst` | Sell-side analyst |
| `operator` | Conference call operator |
| `ir` / `other` | IR host or unclassified |

**Q&A split** — transcript divided into `prepared_remarks` (CEO/CFO opening) and `qa` (full analyst exchange). Split triggers on phrases like "Questions and Answers", "Q&A Session", or an operator line containing "your first question".

### Step 3 — Text cleaning

Applied to all sources (per speaker turn for transcripts, after structural parsing):

| Stripped | Why |
|---|---|
| Repeated page headers/footers | PDF artefacts, zero semantic value |
| Safe-harbour / forward-looking-statement boilerplate | Identical ~500-token block in every filing |
| Legal disclaimer blocks | Same |
| Runs of 3+ blank lines | Formatting noise |

Preserved: original casing, punctuation, numbers, paragraph structure.

### Step 4 — Section tagging (reports only)

Regex patterns match known MD&A headings (`Management's Discussion and Analysis`, `Business Segment Results`, `Canadian Personal Banking`, `Risk Factors`, `Liquidity and Capital`, etc.) so each chunk stays topically coherent. When no heading matches the entire document becomes a single `body` section.

### Step 5 — Semantic chunking

| Parameter | Value |
|---|---|
| Tokeniser | `tiktoken` — `cl100k_base` (GPT-4 / Claude compatible) |
| Target chunk size | 1 800 tokens |
| Hard maximum | 4 000 tokens |
| Split priority | section boundary → paragraph break → sentence break |
| Minimum merge threshold | chunks < ~200 tokens merged into the preceding chunk |

Output — one JSONL per source type in `data/processed/chunks/`:

| File | Documents | Chunks | Tokens |
|---|---:|---:|---:|
| `reports_40f.jsonl` | 5 | 1 023 | ~1.38 M |
| `reports_quarterly.jsonl` | 15 | 1 086 | ~1.48 M |
| `news.jsonl` | 688 | 1 019 | ~1.18 M |
| `transcripts.jsonl` | 20 | 1 363 | ~250 K |
| **Total** | **728** | **4 491** | **~4.29 M** |

> Transcript average is 183 tokens/chunk because each speaker turn is its own chunk. Short participant-header stubs (< 50 tokens) are filtered out at Step 2 before LLM scoring.

---

## Feature engineering

`src/features/build_daily_features.py` merges all processed sources into a single daily feature table:

1. **Price features** — rolling returns, volatility, volume z-scores, price-volume correlation, 52-week distance, sector-relative momentum
2. **Macro features** — yield curve slope, 10Y level, VIX vol, DXY, gold, USD/CAD
3. **News flow features** — 30-day rolling sentiment mean and article count, days since last article
4. **NLP event features** — 11 LLM-scored signals (CEO/CFO tone, guidance, AML pressure, topic entropy, etc.) forward-filled from earnings call date and exponentially decayed (half-life 20 trading days)
5. **Timing features** — days since last earnings call, earnings-week flag

Output: `data/processed/features/model_features_daily.parquet` (~1 285 trading-day rows, 35 feature columns used in Step 3).

---

## Provenance

`data/manifest.csv` has one row per artifact with URL, retrieval timestamp (UTC), SHA-256 checksum, and record count. `src/utils/manifest.py` writes to it automatically when each collector runs.

---

## How to use

All scripts are run from the **project root** directory. They self-add the project root to `sys.path`, so no `PYTHONPATH` changes are needed.

### Prerequisites

```bash
# From project root
pip install -r requirements.txt
```

A `.env` file at the project root is required for scripts that call external APIs (e.g. OpenAI for LLM annotation in Step 2). Data collection itself has no API key requirements — all sources are public.

### Step-by-step execution order

Run the scripts in the order below. Each step produces output that the next step depends on.

#### 1. Collect raw data

```bash
# Daily prices (TD, peers, benchmarks, FX) — ~seconds
python step1_data_collection/src/collectors/prices.py

# Macro series from Bank of Canada Valet — ~seconds
python step1_data_collection/src/collectors/macro.py

# Earnings call transcript PDFs from TD IR — ~minutes (downloads PDFs)
python step1_data_collection/src/collectors/transcripts.py

# Form 40-F and quarterly report PDFs from TD IR — ~minutes
python step1_data_collection/src/collectors/td_ir_reports.py

# Press releases from TD Newsroom — ~minutes (crawls 688 articles)
python step1_data_collection/src/collectors/newsroom.py
```

Each collector is idempotent — re-running skips already-downloaded files and updates `data/manifest.csv`.

#### 2. Build LLM-ready chunks

```bash
# Extracts text, cleans, parses structure, and chunks all sources → data/processed/chunks/
python step1_data_collection/src/preprocess/build_chunks.py
```

Output: four JSONL files in `data/processed/chunks/` (`transcripts.jsonl`, `reports_40f.jsonl`, `reports_quarterly.jsonl`, `news.jsonl`). These are consumed by Step 2 for LLM annotation.

#### 3. Build the daily feature table

```bash
# Merges all processed sources → data/processed/features/model_features_daily.parquet
python step1_data_collection/src/features/build_daily_features.py
```

This requires the LLM annotation outputs from Step 2 (`data/processed/llm_annotations/`) to be present. If Step 2 has not been run yet, NLP event features will be zero-filled.

#### 4. Validate data quality

```bash
# Runs coverage, integrity, and point-in-time sanity checks on the feature table
python step1_data_collection/src/features/data_quality_check.py
```

Expected output summary:
```
Rows:             1,285
Columns total:    171  (1 date + 169 features + 1 target)
Date range:       2021-02-25 → 2026-04-09
Total missing cells: 0 / 217,165  (0.00%)
Constant columns: 0
```

### Validation results

All 11 scripts were validated on 2026-04-22 against the current dataset:

| Script | Status | Notes |
|---|---|---|
| `collectors/prices.py` | ✓ | 9 tickers, ~1 327 trading days |
| `collectors/macro.py` | ✓ | 7 BoC Valet series |
| `collectors/transcripts.py` | ✓ | 20 quarterly transcripts |
| `collectors/td_ir_reports.py` | ✓ | 5 × 40-F + 15 × quarterly reports |
| `collectors/newsroom.py` | ✓ | 688 press releases |
| `preprocess/cleaning.py` | ✓ | Module import |
| `preprocess/chunking.py` | ✓ | Module import |
| `preprocess/build_chunks.py` | ✓ | 4 491 chunks, ~4.29 M tokens |
| `preprocess/llm_cache.py` | ✓ | Module import |
| `features/build_daily_features.py` | ✓ | 1 285 rows × 171 cols |
| `features/data_quality_check.py` | ✓ | 0 missing values, 0 constant columns |
