"""Collect TD quarterly earnings-call transcripts from the TD Investor Relations site.

Scope: only the canonical quarterly earnings calls (four per fiscal year). Industry
conferences and one-off special-event calls are intentionally excluded from the
MVP corpus.

Source: the per-year quarterly-results page:
  https://www.td.com/.../financial-reports/quarterly-results/quarterly-results-{year}

URL naming for the transcript PDF drifts across years (2021-Q1_Transcript_F_EN.pdf,
q3-2024-transcript-en.pdf, 2025-q1-td-transcript.pdf, etc.), so we crawl the HTML
and harvest every PDF link whose URL contains "transcript" and a qN-YYYY token.

Output:
  - data/raw/transcripts/{yyyy-mm-dd}_{slug}.pdf
  - data/raw/transcripts/{yyyy-mm-dd}_{slug}.txt (extracted text)
  - data/raw/transcripts/index.parquet
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    TRANSCRIPTS_DIR,
    WINDOW_END,
    WINDOW_START,
    fiscal_quarter,
)
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transcripts")

IR_BASE = "https://www.td.com"
QUARTERLY_RESULTS_URL = (
    IR_BASE
    + "/ca/en/about-td/for-investors/investor-relations/financial-information"
    + "/financial-reports/quarterly-results/quarterly-results-{year}"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (TD Analysis MVP)"}
REQUEST_SLEEP_SEC = 0.3

DATE_IN_URL = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
_YEAR_Q_PAT = re.compile(r"(\d{4})[-_/]q([1-4])|q([1-4])[-_](\d{4})", re.I)

# Bank-industry conferences that occasionally get dropped into the
# quarterly-results folder. We exclude them here so that the corpus stays
# strictly TD earnings calls.
INDUSTRY_CONF_HINTS = (
    "barclays", "scotia", "rbc", "nbf", "national-bank", "bmo", "cibc",
    "goldman", "morgan-stanley", "cowen", "ubs", "jefferies", "fireside",
    "rbccm", "rbfig",
)


def _infer_date(url: str) -> str | None:
    m = DATE_IN_URL.search(url)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{mo}-{d}"
    return None


def _slug_from_url(url: str) -> str:
    name = Path(url).stem
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return name[:80]


def _fetch(url: str) -> requests.Response:
    time.sleep(REQUEST_SLEEP_SEC)
    return requests.get(url, headers=HEADERS, timeout=30)


def _discover_quarterly(year: int) -> list[dict]:
    """Scrape the quarterly-results-YYYY page for earnings-call transcripts.

    Each year page lists 4 quarters' worth of artifacts; we keep only PDFs
    that (a) have "transcript" in the URL, (b) are not bank-industry
    conference transcripts, and (c) have an explicit qN-YYYY token.
    """
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
        if ".pdf" not in low or "transcript" not in low:
            continue
        if "/fr/" in low or "-fr/" in low or "_fr/" in low:
            continue
        if any(h in low for h in INDUSTRY_CONF_HINTS):
            continue

        m = _YEAR_Q_PAT.search(low)
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
                "link_text": a.get_text(strip=True),
                "inferred_date": _infer_date(full_url),
                "source_page_year": year,
                "td_fiscal_year": fy,
                "td_fiscal_quarter_hint": f"FY{fy}-Q{qn}",
            }
        )
    return out


# Match typical "February 28, 2022" / "Feb 28, 2022" dates near the top of the PDF
_CALL_DATE_PAT = re.compile(
    r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>20\d{2})",
    re.I,
)
# Same, but tolerates zero whitespace (for letter-spaced cover pages that we've
# fully collapsed, e.g. "MARCH02,2023")
_CALL_DATE_PAT_TIGHT = re.compile(
    r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?(?P<day>\d{1,2}),?(?P<year>20\d{2})",
    re.I,
)


def _infer_call_date_from_text(text: str) -> str | None:
    """Grab the first plausible call date from the opening pages of a transcript.

    TD's transcript cover pages are sometimes rendered with letter-spacing
    ("M A R C H  0 2 ,  2 0 2 3"), so we try both the vanilla regex and a
    whitespace-stripped fallback before giving up.
    """
    head = text[:4000]
    m = _CALL_DATE_PAT.search(head)
    if m:
        try:
            return pd.to_datetime(f"{m.group('mon')} {m.group('day')} {m.group('year')}").date().isoformat()
        except Exception:
            pass
    head_tight = re.sub(r"\s+", "", head)
    m = _CALL_DATE_PAT_TIGHT.search(head_tight)
    if m:
        try:
            return pd.to_datetime(f"{m.group('mon')} {m.group('day')} {m.group('year')}").date().isoformat()
        except Exception:
            pass
    return None


def _extract_text(pdf_path: Path) -> str:
    """Light text extraction with pdfplumber."""
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
    return "\n\n".join(pages)


def collect() -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    years = list(range(WINDOW_START.year, WINDOW_END.year + 1))
    all_items: list[dict] = []
    for y in years:
        items = _discover_quarterly(y)
        log.info("quarterly-results-%d: %d transcript links", y, len(items))
        all_items.extend(items)

    # Dedupe by URL
    by_url: dict[str, dict] = {}
    for it in all_items:
        by_url.setdefault(it["url"], it)
    unique = list(by_url.values())

    # Filter to window by inferred_date when we have it; else rely on source_page_year
    in_window = []
    for it in unique:
        d = it.get("inferred_date")
        if d is None:
            py = it["source_page_year"]
            if py < WINDOW_START.year or py > WINDOW_END.year:
                continue
        else:
            dt = pd.to_datetime(d).date()
            if dt < WINDOW_START or dt > WINDOW_END:
                continue
        in_window.append(it)

    log.info("Discovered %d quarterly-call transcript PDFs in window", len(in_window))

    records = []
    for it in in_window:
        url = it["url"]
        date_str = it.get("inferred_date") or f"{it['source_page_year']}-01-01"
        slug = _slug_from_url(url)
        pdf_path = TRANSCRIPTS_DIR / f"{date_str}_{slug}.pdf"
        txt_path = TRANSCRIPTS_DIR / f"{date_str}_{slug}.txt"

        # If a prior run renamed this PDF to a better date prefix, reuse it
        if not pdf_path.exists():
            existing = sorted(TRANSCRIPTS_DIR.glob(f"*_{slug}.pdf"))
            if existing:
                pdf_path = existing[0]
                txt_path = pdf_path.with_suffix(".txt")
                date_str = pdf_path.stem.split("_", 1)[0]

        if not pdf_path.exists():
            r = _fetch(url)
            if r.status_code != 200:
                log.warning("download failed %s -> %d", url, r.status_code)
                continue
            pdf_path.write_bytes(r.content)

        if not txt_path.exists():
            try:
                text = _extract_text(pdf_path)
                txt_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                log.warning("text extraction failed for %s: %s", pdf_path.name, exc)
                text = ""
        else:
            text = txt_path.read_text(encoding="utf-8")

        # Refine date from PDF content when URL didn't give us one
        if it.get("inferred_date") is None and text:
            better = _infer_call_date_from_text(text)
            if better:
                new_pdf = TRANSCRIPTS_DIR / f"{better}_{slug}.pdf"
                new_txt = TRANSCRIPTS_DIR / f"{better}_{slug}.txt"
                if new_pdf != pdf_path and not new_pdf.exists():
                    pdf_path.rename(new_pdf)
                    txt_path.rename(new_txt)
                    pdf_path, txt_path = new_pdf, new_txt
                date_str = better

        try:
            dt = pd.to_datetime(date_str).date()
            fq = fiscal_quarter(dt)
        except Exception:
            fq = ""

        log.info("  %s [%s] %d chars", date_str, fq, len(text))

        record_artifact(
            source="td_ir",
            artifact_type="transcript",
            identifier=pdf_path.stem,
            path=pdf_path,
            url=url,
            notes=f"quarterly_call; {fq}; hint={it.get('td_fiscal_quarter_hint', '')}",
        )
        record_artifact(
            source="td_ir",
            artifact_type="transcript_text",
            identifier=pdf_path.stem,
            path=txt_path,
            url=url,
            record_count=len(text.splitlines()),
            notes="extracted with pdfplumber",
        )

        records.append(
            {
                "date": date_str,
                "fiscal_quarter": fq,
                "td_fiscal_quarter_hint": it.get("td_fiscal_quarter_hint"),
                "url": url,
                "pdf_path": str(pdf_path.relative_to(TRANSCRIPTS_DIR.parent.parent)),
                "txt_path": str(txt_path.relative_to(TRANSCRIPTS_DIR.parent.parent)),
                "char_count": len(text),
            }
        )

    if records:
        idx = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        idx_path = TRANSCRIPTS_DIR / "index.parquet"
        idx.to_parquet(idx_path, index=False)
        log.info("Wrote transcripts index rows=%d -> %s", len(idx), idx_path)
        record_artifact(
            source="td_ir",
            artifact_type="transcript_index",
            identifier="transcripts_index",
            path=idx_path,
            url=IR_BASE,
            record_count=len(idx),
            notes="index of TD quarterly earnings-call transcripts",
        )


if __name__ == "__main__":
    collect()
