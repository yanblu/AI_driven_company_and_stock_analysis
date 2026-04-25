"""Build LLM-ready chunk JSONL files from the raw corpus.

Outputs one JSONL per source type under step1_data_collection/data/chunks/:
  - news.jsonl                (TD newsroom press releases)
  - reports_40f.jsonl         (annual 40-F filings from TD IR, section-tagged)
  - reports_quarterly.jsonl   (quarterly Report to Shareholders from TD IR, section-tagged)
  - transcripts.jsonl         (TD quarterly earnings-call transcripts, speaker + Q&A split)

Each chunk record follows the schema defined in src/preprocess/chunking.py.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocess.chunking import make_chunk_records  # noqa: E402
from src.preprocess.cleaning import (  # noqa: E402
    clean_text,
    parse_transcript,
    tag_sections,
)
from src.utils.config import (  # noqa: E402
    CHUNKS_DIR,
    NEWS_DIR,
    TD_IR_REPORTS_DIR,
    TRANSCRIPTS_DIR,
)
from src.utils.manifest import record_artifact  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_chunks")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# News (TD newsroom)
# -----------------------------------------------------------------------------

def build_news_chunks() -> list[dict]:
    idx_path = NEWS_DIR / "index.parquet"
    if not idx_path.exists():
        log.warning("news index missing, skipping")
        return []

    idx = pd.read_parquet(idx_path)
    records: list[dict] = []
    for _, row in idx.iterrows():
        json_path = NEWS_DIR.parent.parent / row["json_path"]
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        headline = payload.get("headline", "")
        body = payload.get("body", "")
        body = clean_text(body)
        if not body:
            continue

        text_with_headline = f"{headline}\n\n{body}" if headline else body

        extra = {
            "headline": headline,
            "url": payload.get("url", ""),
        }
        for rec in make_chunk_records(
            source_type="news",
            doc_id=row["slug"][:80],
            date=str(row["date"]),
            fiscal_quarter=str(row.get("fiscal_quarter", "")),
            section="press_release",
            speaker=None,
            text=text_with_headline,
            extra_meta=extra,
        ):
            records.append(rec)
    return records


# -----------------------------------------------------------------------------
# TD IR reports (shared helper)
# -----------------------------------------------------------------------------

def _build_td_report_chunks(report_type: str, source_type: str) -> list[dict]:
    """Load td_ir_reports index and emit chunks for the given report_type.

    report_type: "quarterly" | "annual_40f"
    source_type: chunk-file stem used in doc_id / output JSONL
    """
    idx_path = TD_IR_REPORTS_DIR / "index.parquet"
    if not idx_path.exists():
        log.warning("td_ir_reports index missing, skipping")
        return []

    idx = pd.read_parquet(idx_path)
    sub = idx[idx["report_type"] == report_type]
    records: list[dict] = []

    for _, row in sub.iterrows():
        txt_path = TD_IR_REPORTS_DIR.parent.parent / row["txt_path"]
        if not txt_path.exists():
            continue
        try:
            raw = txt_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("read failed %s: %s", txt_path, exc)
            continue
        text = clean_text(raw)
        if not text or len(text) < 500:
            continue

        sections = tag_sections(text)
        if len(sections) <= 1:
            sections = [{"section": "body", "text": text}]

        fy_hint = row["td_fiscal_quarter_hint"]
        doc_stem = Path(row["txt_path"]).stem
        total_chars = sum(len(s["text"]) for s in sections)
        log.info(
            "%s %s [%s] sections=%d chars=%d",
            source_type,
            row["date"],
            fy_hint,
            len(sections),
            total_chars,
        )

        for sec in sections:
            extra = {
                "td_fiscal_year": int(row["td_fiscal_year"]),
                "td_fiscal_quarter_hint": fy_hint,
                "report_type": report_type,
                "url": row["url"],
            }
            for rec in make_chunk_records(
                source_type=source_type,
                doc_id=f"{doc_stem}:{sec['section'][:40]}",
                date=str(row["date"]),
                fiscal_quarter=str(row.get("calendar_fiscal_quarter", "")),
                section=sec["section"],
                speaker=None,
                text=sec["text"],
                extra_meta=extra,
            ):
                records.append(rec)
    return records


def build_reports_40f_chunks() -> list[dict]:
    return _build_td_report_chunks("annual_40f", "reports_40f")


def build_reports_quarterly_chunks() -> list[dict]:
    return _build_td_report_chunks("quarterly", "reports_quarterly")


# -----------------------------------------------------------------------------
# Transcripts
# -----------------------------------------------------------------------------

def build_transcript_chunks() -> list[dict]:
    idx_path = TRANSCRIPTS_DIR / "index.parquet"
    if not idx_path.exists():
        return []
    idx = pd.read_parquet(idx_path)
    records: list[dict] = []

    for _, row in idx.iterrows():
        txt_path = TRANSCRIPTS_DIR.parent.parent / row["txt_path"]
        if not txt_path.exists():
            continue
        raw = txt_path.read_text(encoding="utf-8")
        parsed = parse_transcript(raw)

        doc_id = Path(row["txt_path"]).stem
        for section_name, turns in parsed["sections"].items():
            for ti, turn in enumerate(turns):
                speaker_meta = {
                    "name": turn.get("speaker_name", ""),
                    "role": turn.get("role", "other"),
                    "affiliation": turn.get("affiliation", ""),
                }
                extra = {
                    "td_fiscal_quarter_hint": row.get("td_fiscal_quarter_hint", ""),
                    "url": row.get("url", ""),
                }
                turn_text = clean_text(turn["text"])
                if not turn_text:
                    continue
                for rec in make_chunk_records(
                    source_type="transcripts",
                    doc_id=f"{doc_id}:{section_name}:{ti:03d}",
                    date=str(row["date"]),
                    fiscal_quarter=str(row.get("fiscal_quarter", "")),
                    section=section_name,
                    speaker=speaker_meta,
                    text=turn_text,
                    extra_meta=extra,
                ):
                    records.append(rec)
    return records


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        ("news", build_news_chunks),
        ("reports_40f", build_reports_40f_chunks),
        ("reports_quarterly", build_reports_quarterly_chunks),
        ("transcripts", build_transcript_chunks),
    ]
    summary: list[dict] = []
    for name, fn in builders:
        log.info("building %s chunks...", name)
        records = fn()
        out = CHUNKS_DIR / f"{name}.jsonl"
        _write_jsonl(out, records)
        total_tokens = sum(r.get("token_count", 0) for r in records)
        log.info(
            "wrote %s chunks=%d tokens=%d -> %s",
            name,
            len(records),
            total_tokens,
            out,
        )
        summary.append(
            {
                "source_type": name,
                "chunks": len(records),
                "tokens": total_tokens,
                "path": str(out.relative_to(CHUNKS_DIR.parent.parent)),
            }
        )
        record_artifact(
            source="preprocess",
            artifact_type="chunks",
            identifier=name,
            path=out,
            record_count=len(records),
            notes=f"total_tokens={total_tokens}",
        )

    summary_df = pd.DataFrame(summary)
    log.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
