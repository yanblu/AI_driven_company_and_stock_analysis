"""Collect TD press releases / news from stories.td.com (formerly newsroom.td.com).

Layout:
  - Index: https://stories.td.com/ca/en/news (and /ca/en/news/p{N} for paging)
  - Each page has ~12 article cards; the first 1-2 are "featured" duplicates
    that appear on every page; the remainder are page-specific.
  - Articles: https://stories.td.com/ca/en/news/YYYY-MM-DD-{slug}

We walk pages sequentially and stop once all newly-discovered articles fall
before WINDOW_START. For each article we save:
  - raw/news/{yyyy-mm-dd}_{slug}.html  (full article page)
  - raw/news/{yyyy-mm-dd}_{slug}.json  (headline + body text + metadata)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import (  # noqa: E402
    NEWS_DIR,
    WINDOW_END,
    WINDOW_START,
    fiscal_quarter,
)
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("newsroom")

BASE = "https://stories.td.com"
INDEX_URL = f"{BASE}/ca/en/news"
PAGE_URL = f"{BASE}/ca/en/news/p{{page}}"

HEADERS = {"User-Agent": "Mozilla/5.0 (TD Analysis MVP)"}
REQUEST_SLEEP_SEC = 0.25

ARTICLE_RE = re.compile(r"/ca/en/news/(\d{4}-\d{2}-\d{2})-([^/?#]+)")
MAX_PAGES = 80  # hard safety cap; ~800 articles


def _fetch(url: str) -> requests.Response:
    time.sleep(REQUEST_SLEEP_SEC)
    return requests.get(url, headers=HEADERS, timeout=30)


def _article_links_on_page(url: str) -> list[tuple[str, str, str]]:
    """Return [(full_url, date_str, slug), ...] for article links found on page."""
    r = _fetch(url)
    if r.status_code != 200:
        log.warning("page %s -> %d", url, r.status_code)
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = ARTICLE_RE.search(href)
        if not m:
            continue
        if href.startswith("/"):
            full = BASE + href
        else:
            full = href
        if full in seen:
            continue
        seen.add(full)
        out.append((full, m.group(1), m.group(2)))
    return out


def _parse_article(html: str) -> dict:
    """Extract headline + body + canonical date from an article page."""
    soup = BeautifulSoup(html, "html.parser")

    headline = ""
    h = soup.find(["h1"])
    if h:
        headline = h.get_text(strip=True)

    # Date: prefer a <time> tag, else class-based
    date_str = ""
    t = soup.find("time")
    if t:
        date_str = t.get("datetime") or t.get_text(strip=True)
    if not date_str:
        for el in soup.find_all(class_=re.compile("date", re.I)):
            txt = el.get_text(strip=True)
            if txt:
                date_str = txt
                break

    # Body: join paragraphs inside the main article container
    body_parts: list[str] = []
    article_el = soup.find("article") or soup.find(class_=re.compile("article|story", re.I))
    if article_el:
        for p in article_el.find_all(["p", "li", "h2", "h3"]):
            t2 = p.get_text(" ", strip=True)
            if t2:
                body_parts.append(t2)
    else:
        for p in soup.find_all("p"):
            t2 = p.get_text(" ", strip=True)
            if t2:
                body_parts.append(t2)

    return {
        "headline": headline,
        "date_raw": date_str,
        "body": "\n\n".join(body_parts),
    }


def collect() -> None:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    discovered: dict[str, tuple[str, str]] = {}  # url -> (date, slug)
    stopped_reason = "max_pages"

    for page in range(1, MAX_PAGES + 1):
        url = INDEX_URL if page == 1 else PAGE_URL.format(page=page)
        links = _article_links_on_page(url)
        if not links:
            stopped_reason = "empty_page"
            break

        # Count NEW, in-window items on this page (excluding featured duplicates)
        new_in_window = 0
        all_before_window = True
        for full, d, slug in links:
            try:
                dt = date.fromisoformat(d)
            except Exception:
                continue
            if dt < WINDOW_START or dt > WINDOW_END:
                continue
            all_before_window = False
            if full not in discovered:
                discovered[full] = (d, slug)
                new_in_window += 1

        log.info("page %d: links=%d new_in_window=%d total=%d", page, len(links), new_in_window, len(discovered))

        # Stop once every article on the page is older than WINDOW_START.
        # (Featured items at top of page can be newer, so we check all items.)
        if all_before_window and len(discovered) > 12:
            stopped_reason = "window_exhausted"
            break

    log.info("Discovery done: %d articles, stopped=%s", len(discovered), stopped_reason)

    records: list[dict] = []
    for url, (d, slug) in sorted(discovered.items(), key=lambda kv: kv[1][0]):
        safe_slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-")[:80]
        html_path = NEWS_DIR / f"{d}_{safe_slug}.html"
        json_path = NEWS_DIR / f"{d}_{safe_slug}.json"

        if not html_path.exists():
            r = _fetch(url)
            if r.status_code != 200:
                log.warning("article %s -> %d", url, r.status_code)
                continue
            html_path.write_text(r.text, encoding="utf-8")
        html = html_path.read_text(encoding="utf-8")
        parsed = _parse_article(html)

        try:
            dt = date.fromisoformat(d)
            fq = fiscal_quarter(dt)
        except Exception:
            fq = ""

        payload = {
            "url": url,
            "date": d,
            "fiscal_quarter": fq,
            "slug": slug,
            **parsed,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        record_artifact(
            source="td_newsroom",
            artifact_type="news_html",
            identifier=html_path.stem,
            path=html_path,
            url=url,
            notes=f"{fq}; {parsed.get('headline', '')[:80]}",
        )
        record_artifact(
            source="td_newsroom",
            artifact_type="news_json",
            identifier=json_path.stem,
            path=json_path,
            url=url,
            record_count=len(parsed.get("body", "").splitlines()),
            notes="parsed headline + body",
        )

        records.append(
            {
                "date": d,
                "fiscal_quarter": fq,
                "url": url,
                "slug": slug,
                "headline": parsed.get("headline", ""),
                "body_chars": len(parsed.get("body", "")),
                "json_path": str(json_path.relative_to(NEWS_DIR.parent.parent)),
                "html_path": str(html_path.relative_to(NEWS_DIR.parent.parent)),
            }
        )

    if records:
        idx = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        idx_path = NEWS_DIR / "index.parquet"
        idx.to_parquet(idx_path, index=False)
        log.info("Wrote newsroom index rows=%d -> %s", len(idx), idx_path)
        record_artifact(
            source="td_newsroom",
            artifact_type="news_index",
            identifier="newsroom_index",
            path=idx_path,
            url=INDEX_URL,
            record_count=len(idx),
            notes="all TD newsroom articles in window",
        )


if __name__ == "__main__":
    collect()
