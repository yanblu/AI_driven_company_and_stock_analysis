"""Collect TD's narrative financial reports from TD Investor Relations.

Two document families:

  1. Quarterly Report to Shareholders (10-Q equivalent under TD's Canadian FPI
     regime). Listed on the `quarterly-results-{year}` pages; filename pattern
     varies year-to-year ("2021-Q1_Report_to_Shareholders_F_EN.pdf",
     "2024-q1-reports-shareholders-en.pdf", "q3-2025-report-to-shareholders-en.pdf",
     etc.). Only Q1-Q3 are published as quarterly reports; Q4 is rolled into
     the annual 40-F.

  2. Annual Report on Form 40-F (10-K equivalent). Listed on the 40-F archive
     page. One per fiscal year (TD fiscal year ends October 31).

Output layout:
  data/raw/td_ir_reports/quarterly/{yyyy}_q{n}_report_to_shareholders.pdf
  data/raw/td_ir_reports/quarterly/{yyyy}_q{n}_report_to_shareholders.txt
  data/raw/td_ir_reports/annual/{fy}_form_40f.pdf
  data/raw/td_ir_reports/annual/{fy}_form_40f.txt
  data/raw/td_ir_reports/index.parquet
"""

from __future__ import annotations

import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import (  # noqa: E402
    TD_IR_REPORTS_DIR,
    WINDOW_END,
    WINDOW_START,
    fiscal_quarter,
)
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("td_ir_reports")

IR_BASE = "https://www.td.com"
QUARTERLY_RESULTS_URL = (
    IR_BASE
    + "/ca/en/about-td/for-investors/investor-relations/financial-information"
    + "/financial-reports/quarterly-results/quarterly-results-{year}"
)
FORM_40F_URL = (
    IR_BASE
    + "/ca/en/about-td/for-investors/investor-relations/financial-information"
    + "/annual-report-on-form-40-f"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (TD Analysis MVP)"}
REQUEST_SLEEP_SEC = 0.3

_QUARTERLY_PAT = re.compile(r"(\d{4})[-_]?q([1-4])|q([1-4])[-_](\d{4})", re.I)

# Recognizes "2024", "e-2024-form40f", "E-2024-Form40F", etc.
_FY_40F_PAT = re.compile(r"(?:^|[-_/])(?:e[-_])?(\d{4})[-_]?form[-_]?40[-_]?f", re.I)


# -- Cover-page date inference ------------------------------------------------

_MONTH_ALT = (
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_PAT = re.compile(
    rf"(?P<mon>{_MONTH_ALT})\.?\s+(?P<day>\d{{1,2}}),?\s+(?P<year>20\d{{2}})", re.I
)
_DATE_PAT_TIGHT = re.compile(
    rf"(?P<mon>{_MONTH_ALT})\.?(?P<day>\d{{1,2}}),?(?P<year>20\d{{2}})", re.I
)


def _infer_date_from_text(text: str) -> str | None:
    head = text[:6000]
    m = _DATE_PAT.search(head)
    if m:
        try:
            return pd.to_datetime(
                f"{m.group('mon')} {m.group('day')} {m.group('year')}"
            ).date().isoformat()
        except Exception:
            pass
    head_tight = re.sub(r"\s+", "", head)
    m = _DATE_PAT_TIGHT.search(head_tight)
    if m:
        try:
            return pd.to_datetime(
                f"{m.group('mon')} {m.group('day')} {m.group('year')}"
            ).date().isoformat()
        except Exception:
            pass
    return None


# -- HTTP / IO helpers --------------------------------------------------------


def _fetch(url: str) -> requests.Response:
    time.sleep(REQUEST_SLEEP_SEC)
    return requests.get(url, headers=HEADERS, timeout=60)


def _extract_text(pdf_path: Path) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
    return "\n\n".join(pages)


def _download(url: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    r = _fetch(url)
    if r.status_code != 200:
        log.warning("download failed %s -> %d", url, r.status_code)
        return False
    out_path.write_bytes(r.content)
    return True


def _ensure_text(pdf_path: Path, txt_path: Path) -> str:
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    try:
        text = _extract_text(pdf_path)
    except Exception as exc:
        log.warning("text extraction failed for %s: %s", pdf_path.name, exc)
        text = ""
    txt_path.write_text(text, encoding="utf-8")
    return text


# -- Discovery: quarterly Report to Shareholders ------------------------------


def _discover_quarterly_reports(year: int) -> list[dict]:
    url = QUARTERLY_RESULTS_URL.format(year=year)
    r = _fetch(url)
    if r.status_code != 200:
        log.warning("quarterly-results-%d status=%d", year, r.status_code)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        low = href.lower()
        if ".pdf" not in low:
            continue
        if "shareholders" not in low:
            continue
        if "/fr/" in low or "-fr/" in low or "_fr/" in low:
            continue

        m = _QUARTERLY_PAT.search(low)
        if not m:
            continue
        if m.group(1):
            fy, qn = int(m.group(1)), int(m.group(2))
        else:
            qn, fy = int(m.group(3)), int(m.group(4))

        full_url = urljoin(IR_BASE, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        out.append(
            {
                "url": full_url,
                "td_fiscal_year": fy,
                "td_fiscal_quarter_hint": f"FY{fy}Q{qn}",
                "quarter_num": qn,
                "report_type": "quarterly",
            }
        )
    return out


# -- Discovery: 40-F archive --------------------------------------------------


def _discover_40f() -> list[dict]:
    r = _fetch(FORM_40F_URL)
    if r.status_code != 200:
        log.warning("40-F archive status=%d", r.status_code)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out: list[dict] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        low = href.lower()
        if ".pdf" not in low:
            continue
        m = _FY_40F_PAT.search(low)
        if not m:
            continue
        fy = int(m.group(1))
        full_url = urljoin(IR_BASE, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        out.append(
            {
                "url": full_url,
                "td_fiscal_year": fy,
                "td_fiscal_quarter_hint": f"FY{fy}Q4",
                "quarter_num": 4,
                "report_type": "annual_40f",
            }
        )
    return out


# -- Main ---------------------------------------------------------------------


def collect() -> None:
    quarterly_dir = TD_IR_REPORTS_DIR / "quarterly"
    annual_dir = TD_IR_REPORTS_DIR / "annual"
    quarterly_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(WINDOW_START.year, WINDOW_END.year + 1))

    items: list[dict] = []
    for y in years:
        q = _discover_quarterly_reports(y)
        log.info("quarterly-results-%d: %d shareholder reports", y, len(q))
        items.extend(q)

    forty_f = _discover_40f()
    log.info("40-F archive: %d PDFs (all years)", len(forty_f))
    # Window: only keep FY in window (TD FY ends Oct 31, so FY2021 ends 2021-10-31)
    forty_f = [it for it in forty_f if WINDOW_START.year <= it["td_fiscal_year"] <= WINDOW_END.year]
    log.info("40-F in window: %d PDFs", len(forty_f))
    items.extend(forty_f)

    records: list[dict] = []
    for it in items:
        url = it["url"]
        fy = it["td_fiscal_year"]
        qn = it["quarter_num"]

        if it["report_type"] == "quarterly":
            stem = f"{fy}_q{qn}_report_to_shareholders"
            pdf_path = quarterly_dir / f"{stem}.pdf"
            txt_path = quarterly_dir / f"{stem}.txt"
        else:
            stem = f"fy{fy}_form_40f"
            pdf_path = annual_dir / f"{stem}.pdf"
            txt_path = annual_dir / f"{stem}.txt"

        if not _download(url, pdf_path):
            continue
        text = _ensure_text(pdf_path, txt_path)

        # Prefer cover-page date; fall back to a conservative FY-quarter mapping.
        inferred = _infer_date_from_text(text) if text else None
        if inferred:
            date_str = inferred
        elif it["report_type"] == "annual_40f":
            date_str = f"{fy}-10-31"  # TD fiscal year-end
        else:
            # TD Q1->Jan, Q2->Apr, Q3->Jul; use period-end convention.
            period_end = {1: "01-31", 2: "04-30", 3: "07-31"}[qn]
            date_str = f"{fy}-{period_end}"

        try:
            dt = pd.to_datetime(date_str).date()
            fq_calendar = fiscal_quarter(dt)
        except Exception:
            fq_calendar = it["td_fiscal_quarter_hint"]

        # Skip anything that landed outside the window
        try:
            if dt < WINDOW_START or dt > WINDOW_END:
                continue
        except Exception:
            pass

        log.info(
            "  %s %s [%s] %d chars",
            date_str,
            it["report_type"],
            it["td_fiscal_quarter_hint"],
            len(text),
        )

        record_artifact(
            source="td_ir",
            artifact_type=f"report_{it['report_type']}",
            identifier=stem,
            path=pdf_path,
            url=url,
            notes=f"{it['td_fiscal_quarter_hint']}",
        )
        record_artifact(
            source="td_ir",
            artifact_type=f"report_{it['report_type']}_text",
            identifier=stem,
            path=txt_path,
            url=url,
            record_count=len(text.splitlines()),
            notes="extracted with pdfplumber",
        )

        records.append(
            {
                "date": date_str,
                "td_fiscal_year": fy,
                "td_fiscal_quarter_hint": it["td_fiscal_quarter_hint"],
                "calendar_fiscal_quarter": fq_calendar,
                "report_type": it["report_type"],
                "url": url,
                "pdf_path": str(pdf_path.relative_to(TD_IR_REPORTS_DIR.parent.parent)),
                "txt_path": str(txt_path.relative_to(TD_IR_REPORTS_DIR.parent.parent)),
                "char_count": len(text),
            }
        )

    if records:
        idx = pd.DataFrame(records).sort_values(["date", "td_fiscal_quarter_hint"]).reset_index(drop=True)
        idx_path = TD_IR_REPORTS_DIR / "index.parquet"
        idx.to_parquet(idx_path, index=False)
        log.info("Wrote td_ir_reports index rows=%d -> %s", len(idx), idx_path)
        record_artifact(
            source="td_ir",
            artifact_type="report_index",
            identifier="td_ir_reports_index",
            path=idx_path,
            url=IR_BASE,
            record_count=len(idx),
            notes="index of TD quarterly Report to Shareholders and annual 40-F",
        )


if __name__ == "__main__":
    collect()
