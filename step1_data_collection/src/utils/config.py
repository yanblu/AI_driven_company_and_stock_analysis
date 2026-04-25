"""Path constants and shared configuration for the Step 1 data collection pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

STEP1_DIR = Path(__file__).resolve().parents[2]   # step1_data_collection/
PROJECT_ROOT = STEP1_DIR.parent

DATA_DIR = STEP1_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

PRICES_DIR = RAW_DIR / "prices"
MACRO_DIR = RAW_DIR / "macro"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
TD_IR_REPORTS_DIR = RAW_DIR / "td_ir_reports"
NEWS_DIR = RAW_DIR / "news"

CHUNKS_DIR = DATA_DIR / "chunks"
FEATURES_DIR = DATA_DIR / "features"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Step 2 annotation cache — lives in step2's own folder
LLM_ANNOTATIONS_DIR = PROJECT_ROOT / "step2_llm_analysis" / "data" / "llm_annotations"

# Uniform 5-year window across all data types.
WINDOW_START = date(2021, 1, 1)
WINDOW_END = date.today()

TD_TICKER = "TD.TO"
PEER_TICKERS = ["RY.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO"]
BENCHMARK_TICKERS = ["^GSPTSE", "XFN.TO"]
FX_TICKERS = ["CADUSD=X"]
ALL_TICKERS = [TD_TICKER] + PEER_TICKERS + BENCHMARK_TICKERS + FX_TICKERS


def fiscal_quarter(d: date) -> str:
    """Map a calendar date to TD's fiscal quarter (FY ends Oct 31).

    TD fiscal quarters:
      Q1: Nov 1 - Jan 31  (reported ~late Feb)
      Q2: Feb 1 - Apr 30  (reported ~late May)
      Q3: May 1 - Jul 31  (reported ~late Aug)
      Q4: Aug 1 - Oct 31  (reported ~early Dec)
    """
    month = d.month
    if month in (11, 12):
        return f"FY{d.year + 1}Q1"
    if month in (1,):
        return f"FY{d.year}Q1"
    if month in (2, 3, 4):
        return f"FY{d.year}Q2"
    if month in (5, 6, 7):
        return f"FY{d.year}Q3"
    return f"FY{d.year}Q4"
